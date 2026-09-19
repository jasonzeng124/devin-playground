"""A minimal, readable GRPO implementation using plain PyTorch."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .countdown import format_prompt, verify
from .data import load_puzzles, train_puzzle_stream
from .eval import aggregate_results
from .model_utils import answer_stopping_criteria, load_causal_model, load_tokenizer, resolve_device


@dataclass
class GRPOConfig:
    model: str = ""
    ref_model: str | None = None
    group_size: int = 8
    prompts_per_step: int = 8
    lr: float = 1e-6
    kl_coef: float = 0.02
    temperature: float = 1.0
    max_new_tokens: int = 48
    few_shot: int = 3
    max_steps: int = 100
    eval_every: int = 10
    eval_limit: int = 200
    eval_batch_size: int = 32
    target_solve_rate: float = 1.0
    grad_clip: float = 1.0
    micro_batch_size: int = 16
    curriculum_steps: int = 0
    save_every: int = 0
    seed: int = 0
    device: str = "auto"
    dtype: str | None = None
    out_dir: str = "runs/grpo"
    no_save: bool = False


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _completion_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_width: int):
    start = max(0, prompt_width - 1)
    hidden = model.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    logits = model.get_output_embeddings()(hidden[:, start:-1]).float()
    labels = input_ids[:, start + 1 :]
    token_lp = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    completion_mask = attention_mask[:, start + 1 :].bool()
    return token_lp, completion_mask


def _sample_batch(model, tokenizer, puzzles, config: GRPOConfig, device: torch.device):
    prompts = [format_prompt(puzzle, few_shot=config.few_shot) for puzzle in puzzles]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=config.temperature,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=config.group_size,
            stopping_criteria=answer_stopping_criteria(tokenizer),
        )
    prompt_width = encoded["input_ids"].shape[1]
    completions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
    expanded = [puzzle for puzzle in puzzles for _ in range(config.group_size)]
    rewards = torch.tensor([verify(text, puzzle).reward for text, puzzle in zip(completions, expanded)], dtype=torch.float32, device=device)
    prompt_attention = encoded["attention_mask"].repeat_interleave(config.group_size, dim=0)
    completion = generated[:, prompt_width:]
    completion_attention = torch.ones_like(completion, dtype=prompt_attention.dtype)
    if tokenizer.eos_token_id is not None and completion.shape[1]:
        positions = torch.arange(completion.shape[1], device=device).unsqueeze(0)
        eos_positions = torch.where(
            completion.eq(tokenizer.eos_token_id),
            positions,
            torch.full_like(positions, completion.shape[1]),
        )
        first_eos = eos_positions.min(dim=1, keepdim=True).values
        completion_attention = (positions <= first_eos).to(prompt_attention.dtype)
    attention = torch.cat([prompt_attention, completion_attention], dim=1)
    return generated, attention, completions, expanded, rewards, prompt_width


def grpo_step(model, ref_model, tokenizer, puzzles, config: GRPOConfig, optimizer, device: torch.device) -> dict:
    generated, attention, completions, expanded, rewards, prompt_width = _sample_batch(model, tokenizer, puzzles, config, device)
    groups = rewards.reshape(len(puzzles), config.group_size)
    means = groups.mean(dim=1, keepdim=True)
    stds = groups.std(dim=1, unbiased=False, keepdim=True)
    advantages = (groups - means) / (stds + 1e-4)
    advantages = torch.where(stds > 0, advantages, torch.zeros_like(advantages)).reshape(-1)

    completion_valid = attention[:, prompt_width:].bool()
    token_count = completion_valid.sum().clamp_min(1)
    optimizer.zero_grad(set_to_none=True)
    loss_total = torch.zeros((), device=device)
    kl_total = torch.zeros((), device=device)
    micro = max(1, config.micro_batch_size)
    for i in range(0, generated.shape[0], micro):
        ids, mask, adv = generated[i : i + micro], attention[i : i + micro], advantages[i : i + micro]
        policy_lp, completion_mask = _completion_logprobs(model, ids, mask, prompt_width)
        with torch.no_grad():
            ref_lp, _ = _completion_logprobs(ref_model, ids, mask, prompt_width)
        policy_lp = policy_lp[completion_mask]
        ref_lp = ref_lp[completion_mask]
        repeated_advantages = adv.unsqueeze(1).expand(-1, completion_mask.shape[1])[completion_mask]
        kl_tokens = torch.exp(ref_lp - policy_lp) - (ref_lp - policy_lp) - 1
        policy_loss = -(repeated_advantages * policy_lp).sum() / token_count
        kl = kl_tokens.sum() / token_count
        loss = policy_loss + config.kl_coef * kl
        loss.backward()
        loss_total += loss.detach()
        kl_total += kl.detach()
    loss, kl = loss_total, kl_total
    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    return {
        "mean_reward": rewards.mean().item(),
        "solve_rate_in_batch": (rewards == 1.0).float().mean().item(),
        "malformed_rate": (rewards == -0.1).float().mean().item(),
        "kl": kl.detach().item(),
        "loss": loss.detach().item(),
        "tokens": int(token_count.item()),
        "completions": completions,
        "rewards": rewards.detach().cpu().tolist(),
    }


def _evaluate(model, tokenizer, puzzles, config: GRPOConfig, device: torch.device) -> dict:
    all_completions = []
    with torch.no_grad():
        for start in range(0, len(puzzles), config.eval_batch_size):
            batch = puzzles[start : start + config.eval_batch_size]
            encoded = tokenizer([format_prompt(p, few_shot=config.few_shot) for p in batch], return_tensors="pt", padding=True).to(device)
            generated = model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=answer_stopping_criteria(tokenizer),
            )
            all_completions.extend(tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True))
    result = aggregate_results(all_completions, puzzles)
    result["examples"] = [
        {
            "puzzle": puzzle.to_dict(),
            "completion": completion,
            "reward": verify(completion, puzzle).reward,
        }
        for puzzle, completion in list(zip(puzzles, all_completions))[:2]
    ]
    return result


def run(config: GRPOConfig) -> dict:
    _set_seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    tokenizer = load_tokenizer(config.model)
    model, device = load_causal_model(config.model, config.device, config.dtype)
    ref_name = config.ref_model or config.model
    ref_model, _ = load_causal_model(ref_name, str(device), config.dtype)
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    stream = train_puzzle_stream(config.seed)
    easy_streams = [
        train_puzzle_stream(config.seed + 500_000, n_numbers=2, target_range=(1, 50)),
        train_puzzle_stream(config.seed + 600_000, n_numbers=3, target_range=(1, 50)),
    ]
    mix_rng = random.Random(config.seed)
    eval_puzzles = load_puzzles("data/countdown_eval.jsonl")[: config.eval_limit]
    train_log = (out_dir / "train_log.jsonl").open("w", encoding="utf-8")
    eval_log = (out_dir / "eval_log.jsonl").open("w", encoding="utf-8")
    start = time.perf_counter()
    threshold_time = None
    try:
        for step in range(1, config.max_steps + 1):
            easy_frac = max(0.0, 1.0 - step / config.curriculum_steps) if config.curriculum_steps > 0 else 0.0
            puzzles = []
            for _ in range(config.prompts_per_step):
                u = mix_rng.random()
                source = stream if u >= easy_frac else easy_streams[int(u >= easy_frac / 2)]
                puzzles.append(next(source))
            step_start = time.perf_counter()
            info = grpo_step(model, ref_model, tokenizer, puzzles, config, optimizer, device)
            step_elapsed = time.perf_counter() - step_start
            elapsed = time.perf_counter() - start
            tokens = info.pop("tokens")
            info = {key: value for key, value in info.items() if key not in {"completions", "rewards"}}
            info.update({"step": step, "wall_s": elapsed, "tokens/s": tokens / max(step_elapsed, 1e-9)})
            train_log.write(json.dumps(info) + "\n")
            train_log.flush()
            if config.save_every and step % config.save_every == 0:
                checkpoint = out_dir / f"ckpt_{step}"
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
            if step % config.eval_every == 0 or step == config.max_steps:
                result = _evaluate(model, tokenizer, eval_puzzles, config, device)
                result.update({"step": step, "wall_s": time.perf_counter() - start})
                eval_log.write(json.dumps(result) + "\n")
                eval_log.flush()
                if threshold_time is None and result["pass_rate"] >= config.target_solve_rate:
                    threshold_time = result["wall_s"]
                    break
    finally:
        train_log.close()
        eval_log.close()
    total_wall = time.perf_counter() - start
    if not config.no_save:
        model.save_pretrained(out_dir / "final")
        tokenizer.save_pretrained(out_dir / "final")
    eval_lines = [json.loads(line) for line in (out_dir / "eval_log.jsonl").read_text().splitlines() if line]
    final_pass_rate = eval_lines[-1]["pass_rate"] if eval_lines else 0.0
    result = {
        "final_pass_rate": final_pass_rate,
        "pass_rate": final_pass_rate,
        "time_to_threshold_s": threshold_time,
        "total_steps": len((out_dir / "train_log.jsonl").read_text().splitlines()),
        "total_wall": total_wall,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _config_from_args(args) -> GRPOConfig:
    values = {}
    if args.config:
        values.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    for field in fields(GRPOConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            values[field.name] = value
    return GRPOConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    for field in fields(GRPOConfig):
        arg = "--" + field.name.replace("_", "-")
        if field.name == "no_save":
            parser.add_argument(arg, action="store_true", default=None)
        else:
            parser.add_argument(arg, type=type(field.default) if field.default is not None else str, default=None)
    args = parser.parse_args()
    config = _config_from_args(args)
    if not config.model:
        parser.error("--model is required (or set it in --config)")
    result = run(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
