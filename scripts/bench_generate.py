"""Benchmark rollout generation latency, the dominant cost of a GRPO step.

    python scripts/bench_generate.py --model Qwen/Qwen2.5-0.5B --batch 128 --max-new-tokens 64

Compares HF `generate` with the default dynamic cache against a static cache with
`torch.compile`d decode steps (CUDA graphs), which removes most per-token Python
launch overhead for small models.
"""

from __future__ import annotations

import argparse
import time

import torch

from rlvr_speedrun.countdown import format_prompt, generate_puzzles
from rlvr_speedrun.model_utils import answer_stopping_criteria, load_causal_model, load_tokenizer


def _bench(model, tokenizer, encoded, args, label: str, **generate_kwargs) -> None:
    times, tokens = [], []
    for i in range(args.warmup + args.repeat):
        torch.manual_seed(i)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.max_new_tokens if args.fixed_length else 0,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=args.group_size,
                stopping_criteria=None if args.fixed_length else answer_stopping_criteria(tokenizer),
                **generate_kwargs,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if i >= args.warmup:
            times.append(elapsed)
            tokens.append(out.shape[0] * (out.shape[1] - encoded["input_ids"].shape[1]))
        print(f"{label} iter {i}: {elapsed:.2f}s (out {tuple(out.shape)})", flush=True)
    mean = sum(times) / len(times)
    print(f"{label}: mean {mean:.2f}s over {len(times)} runs, ~{sum(tokens) / sum(times):.0f} tok/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--few-shot", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--fixed-length", action="store_true", help="disable early stopping so runs are comparable")
    parser.add_argument("--mode", choices=["dynamic", "static", "static-manual", "both"], default="both")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    model, device = load_causal_model(args.model, "auto")
    puzzles = generate_puzzles(seed=123, n=args.prompts)
    prompts = [format_prompt(p, few_shot=args.few_shot) for p in puzzles]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    print(f"prompt width {encoded['input_ids'].shape[1]}, rows {args.prompts * args.group_size}", flush=True)

    if args.mode in ("dynamic", "both"):
        _bench(model, tokenizer, encoded, args, "dynamic")
    if args.mode in ("static", "both"):
        # HF auto-compiles the decode forward when a static cache is requested.
        _bench(model, tokenizer, encoded, args, "static(auto-compile)", cache_implementation="static")
    if args.mode == "static-manual":
        model.forward = torch.compile(model.forward, mode="max-autotune-no-cudagraphs", dynamic=False)
        _bench(model, tokenizer, encoded, args, "static+compile(no-cudagraphs)", cache_implementation="static")


if __name__ == "__main__":
    main()
