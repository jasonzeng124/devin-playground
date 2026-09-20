"""A minimal, readable GRPO implementation using plain PyTorch."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import weakref
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from transformers import StaticCache

from .countdown import split_prompt, verify
from .data import load_puzzles, train_puzzle_stream
from .eval import aggregate_results
from .model_utils import answer_stopping_criteria, load_causal_model, load_tokenizer, resolve_device


@dataclass
class GRPOConfig:
    model: str = ""
    ref_model: str | None = None
    group_size: int = 8
    prompts_per_step: int = 8
    oversample: int = 1
    gen_batch_size: int = 0
    compile: bool = False
    pad_to_multiple: int = 0
    adv_norm: str = "std"
    lr: float = 1e-6
    kl_coef: float = 0.02
    temperature: float = 1.0
    max_new_tokens: int = 48
    few_shot: int = 3
    max_steps: int = 100
    eval_every: int = 10
    eval_start_step: int = 0
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


def _completion_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_width: int,
    temperature: float = 1.0,
    prefix_len: int = 0,
):
    """Per-token log-probs (at the sampling temperature) and entropies of completion tokens.

    Positions follow the attention mask (as in `generate`), so padding may sit anywhere.
    If `prefix_len > 0`, columns `[:prefix_len]` are identical and fully attended in
    every row: that prefix is run once and its KV cache broadcast to the batch, which
    is exact under causal attention and skips ~`(rows-1) * prefix_len` tokens of
    forward/backward compute.
    """
    start = max(0, prompt_width - 1)
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
    if 0 < prefix_len <= start:
        prefix = model.base_model(input_ids=input_ids[:1, :prefix_len], position_ids=position_ids[:1, :prefix_len], use_cache=True)
        cache = prefix.past_key_values
        cache.batch_repeat_interleave(input_ids.shape[0])
        hidden = model.base_model(
            input_ids=input_ids[:, prefix_len:],
            attention_mask=attention_mask,
            position_ids=position_ids[:, prefix_len:],
            past_key_values=cache,
            use_cache=True,
        ).last_hidden_state
        hidden = hidden[:, start - prefix_len : -1]
    else:
        hidden = model.base_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).last_hidden_state
        hidden = hidden[:, start:-1]
    logits = model.get_output_embeddings()(hidden).float() / temperature
    labels = input_ids[:, start + 1 :]
    log_z = torch.logsumexp(logits, dim=-1)
    token_lp = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - log_z
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        entropy = log_z - (probs * logits).sum(-1)
    completion_mask = attention_mask[:, start + 1 :].bool()
    return token_lp, completion_mask, entropy


class StaticCachePool:
    """One persistent `StaticCache` per batch size, reset between `generate` calls.

    HF's built-in `cache_implementation="static"` rebuilds the cache whenever the
    batch size changes, and every fresh cache object re-triggers `torch.compile`
    (~25 s on an H100); with alternating rollout/eval batch sizes that happens every
    switch. Reusing the same objects keeps the compiled graphs' guards valid.
    """

    def __init__(self, model):
        self.model = model
        self._caches: dict[int, StaticCache] = {}

    def get(self, rows: int, max_cache_len: int) -> StaticCache:
        cache = self._caches.get(rows)
        if cache is None or cache.max_cache_len < max_cache_len:
            cache = StaticCache(config=self.model.config, max_cache_len=max_cache_len)
            self._caches[rows] = cache
        else:
            cache.reset()
        return cache


def _cache_kwargs(caches: StaticCachePool | None, rows: int, prompt_width: int, config: GRPOConfig) -> dict:
    if caches is None:
        return {}
    return {"past_key_values": caches.get(rows, prompt_width + config.max_new_tokens)}


def _generate_chunked(model, tokenizer, encoded, config: GRPOConfig, caches: StaticCachePool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `group_size` completions per prompt; returns (sequences, finished_at).

    Sampling is from the full tempered softmax (`top_k=0`, `top_p=1.0`; some
    transformers versions default to `top_k=50`) so the sampler matches the policy
    log-probs used in the loss.

    `finished_at[i]` is the generated length of row i: the position at which it
    emitted `</answer>`, or its chunk's generated width if it never did. Tokens at
    or beyond it (forced pads, cross-chunk padding) are excluded from the loss.
    """
    n_prompts = encoded["input_ids"].shape[0]
    chunk = config.gen_batch_size if config.gen_batch_size > 0 else n_prompts
    outputs, finished = [], []
    with torch.no_grad():
        for i in range(0, n_prompts, chunk):
            criteria = answer_stopping_criteria(tokenizer)
            ids = encoded["input_ids"][i : i + chunk]
            outputs.append(
                model.generate(
                    input_ids=ids,
                    attention_mask=encoded["attention_mask"][i : i + chunk],
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    temperature=config.temperature,
                    top_k=0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=config.group_size,
                    stopping_criteria=criteria,
                    **_cache_kwargs(caches, ids.shape[0] * config.group_size, ids.shape[1], config),
                )
            )
            rows, gen_len = outputs[-1].shape[0], outputs[-1].shape[1] - ids.shape[1]
            finished_at = criteria[0].finished_at
            if finished_at is None:
                finished_at = torch.full((rows,), -1, dtype=torch.long, device=outputs[-1].device)
            finished.append(torch.where(finished_at >= 0, finished_at, torch.full_like(finished_at, gen_len)))
    width = max(out.shape[1] for out in outputs)
    padded = [torch.nn.functional.pad(out, (0, width - out.shape[1]), value=tokenizer.pad_token_id) for out in outputs]
    return torch.cat(padded, dim=0), torch.cat(finished, dim=0)


def select_groups(rewards: torch.Tensor, keep: int) -> torch.Tensor:
    """Return indices of the `keep` most informative groups (rows of `rewards`).

    Groups with reward variance carry gradient; among those, groups containing a
    correct solve rank above groups whose only variance is malformed-vs-wrong.
    Zero-variance groups are used last (their advantages are zero).
    """
    n = rewards.shape[0]
    if keep >= n:
        return torch.arange(n, device=rewards.device)
    stds = rewards.std(dim=1, unbiased=False)
    solved = (rewards.max(dim=1).values >= 1.0).float()
    informative = (stds > 0).float()
    score = informative * (1.0 + solved) * 10.0 + stds
    return torch.argsort(score, descending=True, stable=True)[:keep]


def group_advantages(groups: torch.Tensor, adv_norm: str) -> torch.Tensor:
    """Group-relative advantages: `std` is GRPO's per-group z-score, `none` is the
    Dr. GRPO variant (mean-centred only), which removes the difficulty bias that
    up-weights groups where only one or two samples disagree with the rest."""
    centred = groups - groups.mean(dim=1, keepdim=True)
    if adv_norm == "none":
        return centred
    if adv_norm != "std":
        raise ValueError(f"unknown adv_norm {adv_norm!r} (expected 'std' or 'none')")
    stds = groups.std(dim=1, unbiased=False, keepdim=True)
    return torch.where(stds > 0, centred / (stds + 1e-4), torch.zeros_like(centred))


_SPLIT_IS_EXACT: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _encode_prompts(tokenizer, puzzles, config: GRPOConfig, device: torch.device) -> tuple[dict, int]:
    """Tokenize prompts as `[shared prefix][pad...][puzzle suffix]`; returns (encoding, prefix_len).

    Padding between prefix and suffix (rather than on the left) keeps the prefix
    column-aligned across rows. With mask-derived positions this is equivalent to
    left padding for generation, and it lets training reuse the prefix's KV cache.
    Falls back to plain left padding (prefix_len 0) if the tokenizer does not split
    cleanly at the prefix boundary.
    """
    parts = [split_prompt(puzzle, few_shot=config.few_shot) for puzzle in puzzles]
    prefix_ids = tokenizer.encode(parts[0][0])
    suffix_ids = [tokenizer.encode(suffix, add_special_tokens=False) for _, suffix in parts]
    if tokenizer not in _SPLIT_IS_EXACT:
        _SPLIT_IS_EXACT[tokenizer] = tokenizer.encode(parts[0][0] + parts[0][1]) == prefix_ids + suffix_ids[0]
    multiple = max(1, config.pad_to_multiple)
    if not _SPLIT_IS_EXACT[tokenizer]:
        prompts = [prefix + suffix for prefix, suffix in parts]
        kwargs = {"pad_to_multiple_of": multiple} if multiple > 1 else {}
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, **kwargs)
        return {"input_ids": encoded["input_ids"].to(device), "attention_mask": encoded["attention_mask"].to(device)}, 0
    width = len(prefix_ids) + max(len(ids) for ids in suffix_ids)
    width = -(-width // multiple) * multiple
    pad = tokenizer.pad_token_id
    rows, mask = [], []
    for ids in suffix_ids:
        fill = width - len(prefix_ids) - len(ids)
        rows.append(prefix_ids + [pad] * fill + ids)
        mask.append([1] * len(prefix_ids) + [0] * fill + [1] * len(ids))
    return {
        "input_ids": torch.tensor(rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(mask, dtype=torch.long, device=device),
    }, len(prefix_ids)


def _now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _sample_batch(model, tokenizer, puzzles, config: GRPOConfig, device: torch.device, caches: StaticCachePool | None = None):
    encoded, prefix_len = _encode_prompts(tokenizer, puzzles, config, device)
    generated, finished_at = _generate_chunked(model, tokenizer, encoded, config, caches)
    prompt_width = encoded["input_ids"].shape[1]
    completions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
    expanded = [puzzle for puzzle in puzzles for _ in range(config.group_size)]
    rewards = torch.tensor([verify(text, puzzle).reward for text, puzzle in zip(completions, expanded)], dtype=torch.float32, device=device)
    prompt_attention = encoded["attention_mask"].repeat_interleave(config.group_size, dim=0)
    completion = generated[:, prompt_width:]
    positions = torch.arange(completion.shape[1], device=device).unsqueeze(0)
    valid = positions < finished_at.unsqueeze(1)
    if tokenizer.eos_token_id is not None and completion.shape[1]:
        eos_positions = torch.where(
            completion.eq(tokenizer.eos_token_id),
            positions,
            torch.full_like(positions, completion.shape[1]),
        )
        first_eos = eos_positions.min(dim=1, keepdim=True).values
        valid &= positions <= first_eos
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
            valid &= completion.ne(tokenizer.pad_token_id)
    completion_attention = valid.to(prompt_attention.dtype)
    attention = torch.cat([prompt_attention, completion_attention], dim=1)
    return generated, attention, completions, expanded, rewards, prompt_width, prefix_len


def grpo_step(
    model, ref_model, tokenizer, puzzles, config: GRPOConfig, optimizer, device: torch.device, caches: StaticCachePool | None = None
) -> dict:
    t0 = _now(device)
    generated, attention, completions, expanded, rewards, prompt_width, prefix_len = _sample_batch(
        model, tokenizer, puzzles, config, device, caches
    )
    t_sample = _now(device) - t0
    all_rewards = rewards
    all_groups = all_rewards.reshape(len(puzzles), config.group_size)
    informative_groups = int((all_groups.std(dim=1, unbiased=False) > 0).sum().item())
    kept = select_groups(all_groups, config.prompts_per_step)
    if kept.shape[0] < len(puzzles):
        rows = (kept.unsqueeze(1) * config.group_size + torch.arange(config.group_size, device=kept.device)).reshape(-1)
        generated, attention, rewards = generated[rows], attention[rows], rewards[rows]
        row_list = rows.tolist()
        completions = [completions[r] for r in row_list]
        expanded = [expanded[r] for r in row_list]
    groups = rewards.reshape(kept.shape[0], config.group_size)
    advantages = group_advantages(groups, config.adv_norm).reshape(-1)

    completion_valid = attention[:, prompt_width:].bool()
    token_count = completion_valid.sum().clamp_min(1)
    optimizer.zero_grad(set_to_none=True)
    loss_total = torch.zeros((), device=device)
    kl_total = torch.zeros((), device=device)
    entropy_total = torch.zeros((), device=device)
    micro = max(1, config.micro_batch_size)
    for i in range(0, generated.shape[0], micro):
        ids, mask, adv = generated[i : i + micro], attention[i : i + micro], advantages[i : i + micro]
        policy_lp, completion_mask, entropy = _completion_logprobs(model, ids, mask, prompt_width, config.temperature, prefix_len)
        policy_lp = policy_lp[completion_mask]
        entropy_total += entropy[completion_mask].sum()
        repeated_advantages = adv.unsqueeze(1).expand(-1, completion_mask.shape[1])[completion_mask]
        policy_loss = -(repeated_advantages * policy_lp).sum() / token_count
        if ref_model is None:
            kl = torch.zeros((), device=device)
            loss = policy_loss
        else:
            with torch.no_grad():
                ref_lp, _, _ = _completion_logprobs(ref_model, ids, mask, prompt_width, config.temperature, prefix_len)
            ref_lp = ref_lp[completion_mask]
            kl_tokens = torch.exp(ref_lp - policy_lp) - (ref_lp - policy_lp) - 1
            kl = kl_tokens.sum() / token_count
            loss = policy_loss + config.kl_coef * kl
        loss.backward()
        loss_total += loss.detach()
        kl_total += kl.detach()
    loss, kl = loss_total, kl_total
    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    t_update = _now(device) - t0 - t_sample
    return {
        "mean_reward": all_rewards.mean().item(),
        "solve_rate_in_batch": (all_rewards == 1.0).float().mean().item(),
        "malformed_rate": (all_rewards == -0.1).float().mean().item(),
        "informative_groups": informative_groups,
        "kept_groups": int(kept.shape[0]),
        "entropy": (entropy_total / token_count).item(),
        "kl": kl.detach().item(),
        "loss": loss.detach().item(),
        "tokens": int(token_count.item()),
        "t_sample": t_sample,
        "t_update": t_update,
        "completions": completions,
        "rewards": rewards.detach().cpu().tolist(),
    }


def _evaluate(model, tokenizer, puzzles, config: GRPOConfig, device: torch.device, caches: StaticCachePool | None = None) -> dict:
    all_completions = []
    with torch.no_grad():
        for start in range(0, len(puzzles), config.eval_batch_size):
            batch = puzzles[start : start + config.eval_batch_size]
            n_real = len(batch)
            if config.compile and n_real < config.eval_batch_size:
                batch = batch + [batch[0]] * (config.eval_batch_size - n_real)
            encoded, _ = _encode_prompts(tokenizer, batch, config, device)
            generated = model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=answer_stopping_criteria(tokenizer),
                **_cache_kwargs(caches, encoded["input_ids"].shape[0], encoded["input_ids"].shape[1], config),
            )
            all_completions.extend(tokenizer.batch_decode(generated[:n_real, encoded["input_ids"].shape[1] :], skip_special_tokens=True))
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


def _warmup(model, tokenizer, config: GRPOConfig, eval_puzzles, device: torch.device, caches: StaticCachePool) -> None:
    """Untimed warm-up: compile the decode graph for the rollout and eval batch shapes.

    Uses a puzzle stream disjoint from training and performs no optimizer step, and
    restores the RNG state afterwards, so the policy and the seeded sampling sequence
    are unchanged when the clock starts.
    """
    warm_stream = train_puzzle_stream(config.seed + 700_000)
    rollout = [next(warm_stream) for _ in range(config.prompts_per_step * max(1, config.oversample))]
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        _sample_batch(model, tokenizer, rollout, config, device, caches)
        _evaluate(model, tokenizer, eval_puzzles[: config.eval_batch_size], config, device, caches)


def run(config: GRPOConfig) -> dict:
    _set_seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    tokenizer = load_tokenizer(config.model)
    model, device = load_causal_model(config.model, config.device, config.dtype)
    ref_model = None
    if config.kl_coef != 0:
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
    caches = StaticCachePool(model) if config.compile else None
    if caches is not None:
        warm_start = time.perf_counter()
        _warmup(model, tokenizer, config, eval_puzzles, device, caches)
        print(f"warmup (untimed): {time.perf_counter() - warm_start:.1f}s", flush=True)
    start = time.perf_counter()
    threshold_time = None
    try:
        for step in range(1, config.max_steps + 1):
            easy_frac = max(0.0, 1.0 - step / config.curriculum_steps) if config.curriculum_steps > 0 else 0.0
            puzzles = []
            for _ in range(config.prompts_per_step * max(1, config.oversample)):
                u = mix_rng.random()
                source = stream if u >= easy_frac else easy_streams[int(u >= easy_frac / 2)]
                puzzles.append(next(source))
            step_start = time.perf_counter()
            info = grpo_step(model, ref_model, tokenizer, puzzles, config, optimizer, device, caches)
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
            if (step >= config.eval_start_step and step % config.eval_every == 0) or step == config.max_steps:
                result = _evaluate(model, tokenizer, eval_puzzles, config, device, caches)
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
        if field.name in {"no_save", "compile"}:
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
