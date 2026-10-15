# Track A 000: baseline

Plain GRPO from `Qwen/Qwen3-1.7B-Base` (revision pinned in `rlvr_speedrun/benchmark.py`):

* training puzzles from the generator mix, zero-shot r1 prompt, 1,024 new tokens;
* 64 prompts x 8 samples per step, lr 2e-6, no KL, T = 1;
* rollouts with vLLM, updates with gradient checkpointing, `micro_batch_tokens` 8192;
* 290 min of training (`time_budget_s` 17400), leaving ~10 min of the 5 h for start-up and saving.

Scored with free sampling at T = 1 (`constrain: false`, no think budget), 2,400 test puzzles x 16
samples, 1,024 tokens.

This is calibration/sweep_008 `q3_17b_lr2e-6` with `micro_batch_tokens` lowered from 16384 to 8192.
The update is the same, but the larger setting needs 99-108 GB and ran out of memory on the
H100 80GB (the failed attempt is in `extra/oom_0`).

## Results

All seeds ran on MIT Engaging, one NVIDIA H100 80GB HBM3 (node2906), with torch 2.13.0+cu130,
transformers 5.17.0, vLLM 0.30.0. The clock counts all of the GPU wall time except the
in-training dev evals (RULES_v3).

| seed | test ladder | L1 | L2 | L3 | L4 | L5 | L6 | mean tokens | clock (min) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1297 | 0.590 | 0.079 | 0.011 | 0.015 | 0.004 | 0.078 | 371 | 291 |
| 1 | 0.1236 | 0.566 | 0.075 | 0.010 | 0.016 | 0.004 | 0.070 | 379 | 291 |
| 2 | 0.1221 | 0.568 | 0.069 | 0.009 | 0.013 | 0.005 | 0.068 | 403 | 291 |
| 3 | 0.1463 | 0.645 | 0.101 | 0.014 | 0.022 | 0.005 | 0.091 | 420 | 291 |
| 4 | 0.1258 | 0.576 | 0.077 | 0.013 | 0.015 | 0.004 | 0.070 | 373 | 291 |

Mean test ladder score **0.1295** (sd 0.0098, 5 seeds).

## What the model does

It writes a guess-and-check search before answering: it lists candidate expressions, computes
them and compares them with the target. The arithmetic in the search is often wrong ("22 - 7 - 3 - 2 = 14 (this works!)"),
and it sometimes loops on one candidate until it hits the 1,024-token limit. It solves most L1
puzzles (+ and - only) and few at L3-L5.
