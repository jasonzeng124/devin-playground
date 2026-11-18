"""Where a D RL step's sampling time goes (timing only; run on a GPU node, never the login node).

Loads a config (e.g. calibration/research_r96/final_share_mbt65k/config.json) and a checkpoint, builds the vLLM engine as
grpo.run does, and times `VLLMRollout.generate` on fresh training batches: wall time per call, completion lengths, the
constrained processor's CPU time, a cProfile of a few calls (top functions by own time), and the same calls without
the answer constraint as a bound. Prints one JSON line per variant and the profile as text.

    python scripts/research/profile_sampling.py --config <config.json> --model <checkpoint dir> [--reps 20]
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
from rlvr_speedrun.model_utils import load_tokenizer
from rlvr_speedrun.vllm_rollout import VLLMRollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-reps", type=int, default=3)
    args = parser.parse_args()
    names = {f.name for f in fields(grpo.GRPOConfig)}
    loaded = json.loads(Path(args.config).read_text())
    config = replace(grpo.GRPOConfig(**{k: v for k, v in loaded.items() if k in names}), model=args.model)
    device = torch.device("cuda")
    tokenizer = load_tokenizer(config.model)
    engine = VLLMRollout(config.model, 512 + max(config.max_new_tokens, config.eval_max_new_tokens), config.vllm_gpu_memory, config.seed,
                         constrain_answer=config.constrain_answer, constrain_batched=config.constrain_batched)
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
            mask_cpu.append(vllm_constrained.APPLY_TIME)
            vllm_constrained.APPLY_TIME = None
            walls.append(wall)
            mean_len.append(finished.float().mean().item())
            max_len.append(finished.max().item())
        print(json.dumps({"variant": "constrained" if constrained else "unconstrained", "median_wall_s": statistics.median(walls),
                          "mean_wall_s": statistics.mean(walls), "mean_completion_len": statistics.mean(mean_len),
                          "mean_max_len": statistics.mean(max_len), "mask_cpu_s": statistics.mean(mask_cpu),
                          "per_decode_step_ms": 1000 * statistics.mean(w / m for w, m in zip(walls, max_len))}), flush=True)

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
