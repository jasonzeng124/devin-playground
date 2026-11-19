"""Score saved checkpoints on the dev set by both in-loop samplers (vLLM and native), puzzle by puzzle.

A run with `rollout_backend: "native"` is also evaluated by the native sampler, so a dev gap between a native
run and its vLLM twin can come from training or from evaluation. This scores the same weights by both paths:

- greedy, as `grpo._evaluate` does in the loop (`_evaluate`'s own number for each sampler, and one more pass
  that keeps every completion, so the two samplers are compared puzzle by puzzle);
- with `--samples k`, k completions per puzzle at the training temperature, in calls with as many rows as a
  training step's: solve rate, length, distinct completions per prompt, and the mean log-prob and entropy of
  the sampled tokens under the fp32 weights (unmasked), which says whether the two samplers draw from the same
  distribution.

Both engines are built as `grpo.run` builds them (vLLM first, on an empty GPU, then the training model, then
the native sampler from it) and take each checkpoint's weights through `sync`, as they do after a training
step. Only `eval_set` is read (the dev ladder in the configs that use this); nothing is trained or saved.

  python scripts/research/sampler_paths_eval.py --config <config.json> --models <checkpoint dir>... --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlvr_speedrun.countdown import verify  # noqa: E402
from rlvr_speedrun.distributed import SINGLE  # noqa: E402
from rlvr_speedrun.grpo import (  # noqa: E402
    GRPOConfig,
    _encode_prompts,
    _evaluate,
    _load_eval_set,
    _wants_sampler_logprobs,
)
from rlvr_speedrun.model_utils import load_causal_model, load_tokenizer  # noqa: E402

BACKENDS = ("vllm", "native")


def _config(path: str, model: str) -> GRPOConfig:
    loaded = json.loads(Path(path).read_text())
    names = {field.name for field in fields(GRPOConfig)}
    config = GRPOConfig(**{key: value for key, value in loaded.items() if key in names})
    config.model, config.device = model, "cuda"
    return config


def _generate(engine, tokenizer, puzzles, config, device, n, temperature, limit, style):
    """One sampler call as `_evaluate` and `_sample_batch` make it; returns per-row (prompt ids, completion ids) and the texts."""
    encoded, _ = _encode_prompts(tokenizer, puzzles, config, device, style)
    sequences, lengths = engine.generate(encoded, n, temperature, limit, tokenizer.pad_token_id,
                                         puzzles=puzzles if config.constrain_answer else None, think_budget=config.think_budget)
    width = encoded["input_ids"].shape[1]
    rows = []
    for r in range(len(puzzles) * n):
        prompt = encoded["input_ids"][r // n][encoded["attention_mask"][r // n].bool()].tolist()
        rows.append((prompt, sequences[r, width : width + int(lengths[r])].tolist()))
    texts = tokenizer.batch_decode(sequences[: len(puzzles) * n, width:], skip_special_tokens=True)
    return rows, texts


@torch.no_grad()
def _score_under_model(model, rows, pad_id, device, batch_rows=64):
    """Sum of the sampled tokens' log-probs and of the model's entropy at their positions (fp32, no answer mask)."""
    logprob = entropy = 0.0
    tokens = 0
    for start in range(0, len(rows), batch_rows):
        chunk = rows[start : start + batch_rows]
        width = max(len(p) + len(c) for p, c in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        target = torch.zeros((len(chunk), width), dtype=torch.bool)
        for i, (p, c) in enumerate(chunk):
            ids[i, : len(p) + len(c)] = torch.tensor(p + c, dtype=torch.long)
            mask[i, : len(p) + len(c)] = 1
            target[i, len(p) : len(p) + len(c)] = True
        ids, mask, target = ids.to(device), mask.to(device), target.to(device)
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), dim=-1)
        picked = logp[:, :-1].gather(-1, ids[:, 1:, None])[..., 0]
        ent = -(logp[:, :-1].exp() * logp[:, :-1]).sum(-1)
        logprob += picked[target[:, 1:]].sum().item()
        entropy += ent[target[:, 1:]].sum().item()
        tokens += int(target[:, 1:].sum().item())
    return logprob, entropy, tokens


def _by_level(puzzles, values):
    levels: dict[str, list[float]] = {}
    for puzzle, value in zip(puzzles, values):
        levels.setdefault(puzzle.level or "", []).append(value)
    return {lv: sum(v) / len(v) for lv, v in levels.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="a GRPO config.json: prompt, eval set, sampler and vLLM settings")
    parser.add_argument("--models", required=True, nargs="+", help="checkpoint dirs, scored one after the other")
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=0, help="completions per puzzle at the training temperature (0 = greedy only)")
    args = parser.parse_args()
    models = [os.path.expandvars(os.path.expanduser(m)) for m in args.models]
    config = _config(args.config, models[0])
    assert config.eval_set and "test" not in Path(config.eval_set).name, "the dev set only"
    limit = config.eval_max_new_tokens or config.max_new_tokens
    max_len = 512 + max(config.max_new_tokens, config.eval_max_new_tokens)
    # sampled calls have as many rows as a training step's (prompts_per_step x group_size)
    prompts = max(1, config.prompts_per_step * config.group_size // max(1, args.samples))
    tokenizer = load_tokenizer(config.model)
    puzzles = _load_eval_set(config)

    from rlvr_speedrun.vllm_rollout import VLLMRollout

    engines = {"vllm": VLLMRollout(
        config.model, max_len, config.vllm_gpu_memory, config.seed, constrain_answer=config.constrain_answer,
        logprobs=_wants_sampler_logprobs(config), engine_args=json.loads(config.vllm_engine_args) if config.vllm_engine_args else None,
        constrain_batched=config.constrain_batched,
    )}
    out = {"config": args.config, "eval_set": config.eval_set, "n": len(puzzles), "limit": limit, "samples": args.samples,
           "temperature": config.temperature, "constrain_batched_vllm": config.constrain_batched,
           "gpu": torch.cuda.get_device_name(0), "checkpoints": []}
    for path in models:
        t0 = time.time()
        model, device = load_causal_model(path, config.device, "fp32" if config.master_fp32 else config.dtype)
        if "native" not in engines:
            from rlvr_speedrun.fast_rollout import NativeRollout

            engines["native"] = NativeRollout(model, tokenizer, max_len, config.seed, constrain_answer=config.constrain_answer)
        entry: dict = {"model": path, "greedy": {}, "sampled": {}}
        texts: dict[str, list[str]] = {}
        tokens: dict[str, list[list[int]]] = {}
        for name in BACKENDS:
            engine = engines[name]
            engine.sync(model)
            official = _evaluate(model, tokenizer, puzzles, config, device, engine, SINGLE)
            rows, texts[name] = [], []
            for start in range(0, len(puzzles), config.eval_batch_size):
                batch = puzzles[start : start + config.eval_batch_size]
                got, text = _generate(engine, tokenizer, batch, config, device, 1, 0.0, limit, config.eval_prompt_style or config.prompt_style)
                rows += got
                texts[name] += text
            tokens[name] = [c for _, c in rows]
            checks = [verify(t, p) for t, p in zip(texts[name], puzzles)]
            correct = [int(c.correct) for c in checks]
            by_level = _by_level(puzzles, correct)
            entry["greedy"][name] = {
                "evaluate_pass_rate": official["pass_rate"], "evaluate_ladder_score": official.get("ladder_score"),
                "pass_rate": sum(correct) / len(correct), "ladder_score": sum(by_level.values()) / len(by_level),
                "pass_by_level": by_level, "malformed_rate": sum(c.malformed for c in checks) / len(checks),
                "mean_tokens": sum(len(c) for c in tokens[name]) / len(rows),
                "at_limit": sum(len(c) >= limit for c in tokens[name]), "correct": "".join(map(str, correct)),
            }
            if args.samples:
                solved, lengths, distinct, at_limit, malformed = [], 0, 0, 0, 0
                logprob = entropy = 0.0
                scored = 0
                for start in range(0, len(puzzles), prompts):
                    batch = puzzles[start : start + prompts]
                    got, text = _generate(engine, tokenizer, batch, config, device, args.samples, config.temperature,
                                          config.max_new_tokens, config.prompt_style)
                    for i, puzzle in enumerate(batch):
                        mine = text[i * args.samples : (i + 1) * args.samples]
                        results = [verify(t, puzzle) for t in mine]
                        solved.append(sum(r.correct for r in results))
                        malformed += sum(r.malformed for r in results)
                        distinct += len(set(mine))
                    lengths += sum(len(c) for _, c in got)
                    at_limit += sum(len(c) >= config.max_new_tokens for _, c in got)
                    lp, ent, n_tokens = _score_under_model(model, got, tokenizer.pad_token_id, device)
                    logprob, entropy, scored = logprob + lp, entropy + ent, scored + n_tokens
                total = len(puzzles) * args.samples
                rate = _by_level(puzzles, [s / args.samples for s in solved])
                entry["sampled"][name] = {
                    "solve_rate": sum(solved) / total, "ladder_solve_rate": sum(rate.values()) / len(rate),
                    "solve_by_level": rate, "malformed_rate": malformed / total, "mean_tokens": lengths / total,
                    "at_limit": at_limit, "distinct_per_prompt": distinct / len(puzzles),
                    "model_logprob_per_token": logprob / max(1, scored), "model_entropy_per_token": entropy / max(1, scored),
                    "solved": solved,
                }
        a, b = (entry["greedy"][name]["correct"] for name in BACKENDS)
        differ = [i for i in range(len(puzzles)) if texts["vllm"][i] != texts["native"][i]]
        first = []
        for i in differ:
            x, y = tokens["vllm"][i], tokens["native"][i]
            first.append(next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y))))
        entry["greedy"]["paired"] = {
            "identical": len(puzzles) - len(differ), "differ": len(differ),
            "both": sum(u == "1" and v == "1" for u, v in zip(a, b)), "vllm_only": sum(u == "1" and v == "0" for u, v in zip(a, b)),
            "native_only": sum(u == "0" and v == "1" for u, v in zip(a, b)), "neither": sum(u == "0" and v == "0" for u, v in zip(a, b)),
            "first_differing_token": first,
            "examples": [{"index": i, "level": puzzles[i].level, "vllm": texts["vllm"][i], "native": texts["native"][i],
                          "vllm_correct": a[i] == "1", "native_correct": b[i] == "1"} for i in differ[:40]],
        }
        entry["seconds"] = round(time.time() - t0, 1)
        out["checkpoints"].append(entry)
        g, p = entry["greedy"], entry["greedy"]["paired"]
        print(f"{path}: greedy vllm {g['vllm']['pass_rate']:.4f} native {g['native']['pass_rate']:.4f} "
              f"identical {p['identical']}/{len(puzzles)} vllm_only {p['vllm_only']} native_only {p['native_only']}", flush=True)
        if args.samples:
            s = entry["sampled"]
            print("  sampled: " + "; ".join(
                f"{name} solve {s[name]['solve_rate']:.4f} tokens {s[name]['mean_tokens']:.1f} distinct {s[name]['distinct_per_prompt']:.2f} "
                f"logprob {s[name]['model_logprob_per_token']:.4f} entropy {s[name]['model_entropy_per_token']:.4f}" for name in BACKENDS), flush=True)
        tmp = args.out + ".tmp"
        Path(tmp).write_text(json.dumps(out, indent=1) + "\n")
        os.replace(tmp, args.out)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
