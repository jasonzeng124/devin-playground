# Track A 007: A 006 with 96 prompts per step and a longer schedule

A track A record on the H200 board. Mean test ladder score of all five seeds:
**0.4527 ± 0.0113**. Every seed finished within the 5-h budget.

`scripts/validate_record_v3.py` passes. Against the H200 holder, A 006 (0.4123 ± 0.0127), the
one-sided Welch test gives t = 5.31, df 7.9, p = 0.0004, so A 007 replaces A 006 as the track A
holder. The gain is +0.040.

## What ran

The recipe is A 006's, with three changes:

- 96 prompts per step instead of 64 (still 8 samples per prompt).
- The cosine down to 0.1x is counted over 583 steps (`budget_steps` 583) instead of 500.
- Training stops at step 467 (`max_steps` 467) instead of 400. `budget_steps_time_stop` would
  also stop it at 17,400 s of real time.

A 006's recipe, unchanged:

- Qwen3-1.7B-Base.
- Training puzzles come from the ladder-level generators. The dev and test ladders are excluded.
- Answers are constrained to well-formed expressions during rollouts.
- fp32 master weights; lr 1e-6 on the cosine.
- Zero-advantage rows are skipped.
- Truncated outputs get a training reward of 0.
- An entropy floor: coefficient 0.05 while entropy is below 0.1.
- `keep_best_dev`: the run evaluates on the dev ladder every 50 steps from step 50 and at the last
  step, and ends on the weights of its best in-loop dev eval. Those evals and the weight copies
  are on the clock. **This record chose its checkpoint by its own in-loop dev evals**, which
  RULES_v3 allows since v3.4 (2026-11-10). The test ladder is scored once, on the chosen weights.
- Sampler: `constrain: true`, T = 1, no think budget.
- No external data.

It was promoted from research screen `calibration/research_r75/a006_p96_cos583_s467_h200` (H200),
which is this config with `no_save`. Its kept dev evals were 0.450, 0.445 and 0.438, each at least
A 006's line of 0.418 (dev 0.388 + 0.03), which meets the 3-seed promotion rule. The batch size
and schedule came from B200 screens r60 and r63.

The seeds ran on 2026-11-15 and 2026-11-16 on two mit_preemptable H200 SXM workers
(`NVIDIA H200`) on MIT Engaging. Each seed ran `scripts/research/run_item.sh`: first
`scripts/warm_start.sh`, untimed, then `scripts/run_seed_v3.sh`. The software is this repository
at the commits listed in `docs/research/LOG.md`, with `uv.lock`'s package versions.

| seed | node | test ladder | pass@16 | mean tokens | truncated | kept step | kept in-loop dev | last in-loop dev | steps | total_s | within 18,000 s | warm start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | node4602 | 0.4557 | 0.7283 | 690 | 0.51 | 400 | 0.435 | 0.432 | 467 | 15815 | yes | 25 s |
| 1 | node4602 | 0.4391 | 0.6633 | 687 | 0.50 | 350 | 0.427 | 0.402 | 467 | 15146 | yes | 7 s |
| 2 | node4602 | 0.4527 | 0.7183 | 682 | 0.50 | 450 | 0.442 | 0.412 | 467 | 14793 | yes | 10 s |
| 3 | node4602 | 0.4693 | 0.7100 | 677 | 0.50 | 467 | 0.427 | 0.427 | 467 | 15624 | yes | 7 s |
| 4 | node4602 | 0.4464 | 0.7175 | 672 | 0.48 | 467 | 0.413 | 0.413 | 467 | 15048 | yes | 13 s |

- Test ladder score, mean of all five seeds: 0.4527 ± 0.0113. The kept in-loop dev mean is 0.429.
- On the clock, the seeds took 246.6-263.6 min of the 300. A 006 took 139-160 min, so A 007 uses
  more of the budget: 1.5x the prompts per step and 467 steps instead of 400.
- `warm_start.log` lists every warm start, including those of seed 1's preempted attempts.

## What the checkpoint choice did

Three seeds kept an earlier checkpoint than their last one (step 467): seed 0 kept step 400 (dev
0.435 against 0.432 at the end), seed 1 kept step 350 (0.427 against 0.402) and seed 2 kept step
450 (0.442 against 0.412). Seeds 3 and 4 kept their last step. The in-loop dev eval is on 600
puzzles, so a gap of 0.01-0.03 between two evals is within its noise; seed 1's 0.025 is the
largest.

## Preempted attempts

Seed 1 was preempted twice before it ran to the end. Its partial runs are in `extra/`:

- `extra/interrupted_1_1`: node4600, started 2026-11-16 00:34Z, stopped at step 50 (860 s of
  training).
- `extra/interrupted_1_2`: node4301, started 02:10Z, stopped at step 107 (2,749 s of training).

As RULES_v3 allows, each preemption reran the same seed from scratch, and the seed counts once,
with its complete run. No other seed was interrupted.
