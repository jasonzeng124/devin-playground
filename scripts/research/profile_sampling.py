"""Where a D RL step's sampling time goes (timing only; run on a GPU node, never the login node).

Loads a config (e.g. calibration/research_r96/final_share_mbt65k/config.json) and a checkpoint, builds the vLLM engine as
grpo.run does, and times `VLLMRollout.generate` on fresh training batches: wall time per call, completion lengths, the
constrained processor's CPU time, a cProfile of a few calls (top functions by own time), and the same calls without
the answer constraint as a bound. Prints one JSON line per variant and the profile as text. `--backend native` times the
same-process sampler (`fast_rollout.NativeRollout`) instead, and its per-step weight copy (`sync`); with `--floor` it also
times the native decode loop's GPU work alone (forward; plus sampling; plus the per-token host copy), without the
grammar and stop checks. `--prompts-per-step` overrides the config's batch.

    python scripts/research/profile_sampling.py --config <config.json> --model <checkpoint dir> [--reps 20] [--backend native [--floor]]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import statistics
import time
from dataclasses import fields, replace
from pathlib import Path

import torch

from rlvr_speedrun import grpo, vllm_constrained
from rlvr_speedrun.model_utils import load_causal_model, load_tokenizer
from rlvr_speedrun.vllm_rollout import VLLMRollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-reps", type=int, default=3)
    parser.add_argument("--backend", choices=("vllm", "native"), default="vllm")
    parser.add_argument("--no-graphs", action="store_true", help="native: decode without CUDA graphs")
    parser.add_argument("--prompts-per-step", type=int, default=0, help="override the config's prompts_per_step (0 = keep)")
    parser.add_argument("--floor", action="store_true",
                        help="native: also time the decode loop's GPU work alone (see `floor` below); off by default")
    args = parser.parse_args()
    names = {f.name for f in fields(grpo.GRPOConfig)}
    loaded = json.loads(Path(args.config).read_text())
    config = replace(grpo.GRPOConfig(**{k: v for k, v in loaded.items() if k in names}), model=args.model)
    if args.prompts_per_step:
        config = replace(config, prompts_per_step=args.prompts_per_step)
    device = torch.device("cuda")
    tokenizer = load_tokenizer(config.model)
    max_len = 512 + max(config.max_new_tokens, config.eval_max_new_tokens)
    native = args.backend == "native"
    if native:
        from rlvr_speedrun.fast_rollout import NativeRollout

        model, _ = load_causal_model(config.model, "cuda", "fp32")
        engine = NativeRollout(model, tokenizer, max_len, config.seed, constrain_answer=config.constrain_answer,
                               cuda_graphs=not args.no_graphs, factored_grammar=config.factored_grammar)
    else:
        engine = VLLMRollout(config.model, max_len, config.vllm_gpu_memory, config.seed, constrain_answer=config.constrain_answer,
                             constrain_batched=config.constrain_batched)
    stream, _ = grpo._train_streams(config)

    def batch():
        puzzles = [next(stream) for _ in range(config.prompts_per_step)]
        encoded, _ = grpo._encode_prompts(tokenizer, puzzles, config, device)
        return puzzles, encoded

    def call(puzzles, encoded, constrained=True):
        torch.cuda.synchronize()
        start = time.perf_counter()
        seqs, finished = engine.generate(encoded, config.group_size, config.temperature, config.max_new_tokens, tokenizer.pad_token_id,
                                         puzzles=puzzles if constrained else None, think_budget=config.think_budget)
        torch.cuda.synchronize()
        return time.perf_counter() - start, finished

    for constrained in (True, False):
        for _ in range(args.warmup):
            call(*batch(), constrained)
        walls, mean_len, max_len, mask_cpu = [], [], [], []
        for _ in range(args.reps):
            vllm_constrained.APPLY_TIME = 0.0
            wall, finished = call(*batch(), constrained)
            mask_cpu.append(engine.mask_cpu if native else vllm_constrained.APPLY_TIME)
            vllm_constrained.APPLY_TIME = None
            walls.append(wall)
            mean_len.append(finished.float().mean().item())
            max_len.append(finished.max().item())
        print(json.dumps({"backend": args.backend, "variant": "constrained" if constrained else "unconstrained", "median_wall_s": statistics.median(walls),
                          "mean_wall_s": statistics.mean(walls), "mean_completion_len": statistics.mean(mean_len),
                          "mean_max_len": statistics.mean(max_len), "mask_cpu_s": statistics.mean(mask_cpu),
                          "per_decode_step_ms": 1000 * statistics.mean(w / m for w, m in zip(walls, max_len)),
                          "rows": config.prompts_per_step * config.group_size}), flush=True)
        if constrained:
            longest = round(statistics.mean(max_len))

    if native and args.floor:
        # The decode loop's GPU work without the host's per-token part, for `longest` steps (the constrained calls'
        # mean longest completion, so the same key buckets): prefill, then each step's CUDA-graph forward (mode
        # "forward"), plus the sampling kernels and token copy as `generate` runs them ("sample"), plus the per-token
        # copy of the sampled ids to the host that `generate` makes ("sample_sync"). No grammar, no stop check: the
        # gap between "sample" and the constrained calls' per-step time is what a sampler that keeps the grammar
        # and stop checks on the GPU, syncing every k tokens, could at most remove. Timing only.
        def floor(encoded, mode):
            ids, mask = encoded["input_ids"], encoded["attention_mask"]
            p, width = ids.shape
            n = config.group_size
            b = p * n
            rows = -(-b // engine.batch_bucket) * engine.batch_bucket
            engine._ensure_capacity(rows)
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.perf_counter()
                logits = engine._prefill(ids, mask, n, rows)
                torch.cuda.synchronize()
                prefill = time.perf_counter() - start
                start = time.perf_counter()
                for s in range(longest):
                    if mode != "forward":
                        probs = torch.softmax(logits / config.temperature, dim=-1)
                        noise = torch.empty_like(probs).exponential_(generator=engine.generator)
                        sampled = (probs / noise).argmax(-1)
                        if mode == "sample_sync":
                            sampled.tolist()
                        engine.tokens[:b].copy_(sampled)
                    engine.slot.fill_(width + s)
                    keys = min(engine.max_len, -(-(width + s + 1) // engine.key_bucket) * engine.key_bucket)
                    logits = engine._step(rows, keys)[:b]
                    engine.positions[:b] += 1
                torch.cuda.synchronize()
            return prefill, (time.perf_counter() - start) / longest

        for mode in ("forward", "sample", "sample_sync"):
            for _ in range(args.warmup):
                floor(batch()[1], mode)
            timed = [floor(batch()[1], mode) for _ in range(args.reps)]
            print(json.dumps({"backend": "native", "floor": mode, "steps": longest,
                              "rows": config.prompts_per_step * config.group_size,
                              "prefill_ms": 1000 * statistics.median(t[0] for t in timed),
                              "per_decode_step_ms": 1000 * statistics.median(t[1] for t in timed)}), flush=True)

    if native:
        syncs = []
        for _ in range(args.reps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            engine.sync(model)
            torch.cuda.synchronize()
            syncs.append(time.perf_counter() - start)
        print(json.dumps({"backend": "native", "sync_median_s": statistics.median(syncs)}), flush=True)

    profiler = cProfile.Profile()
    batches = [batch() for _ in range(args.profile_reps)]
    profiler.enable()
    for puzzles, encoded in batches:
        call(puzzles, encoded)
    profiler.disable()
    for key in ("tottime", "cumulative"):
        out = io.StringIO()
        pstats.Stats(profiler, stream=out).sort_stats(key).print_stats(35)
        print(f"== cProfile of {args.profile_reps} constrained calls, by {key} ==\n" + out.getvalue(), flush=True)


if __name__ == "__main__":
    main()
