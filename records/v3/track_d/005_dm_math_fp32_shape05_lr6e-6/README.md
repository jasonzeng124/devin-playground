# Track D 005: D 004 with partial credit 0.5 and RL lr 6e-6

D 004's recipe (`records/v3/track_d/004_dm_math_fp32_h200_warm`) with two RL settings changed: the
GRPO learning rate goes from 3e-6 to 6e-6, and wrong answers earn partial credit (`shaping: 0.5`).
Pretraining is unchanged. Mean test ladder score of all five seeds: **0.2150 ± 0.0287**, against
D 004's 0.1552 ± 0.0217. Every seed finished within the 6-h budget.

## What ran

`config.json` is D 004's config with these changes:

- `lr`: 3e-6 → 6e-6.
- `shaping`: 0.0 → 0.5. A well-formed wrong answer adds 0.5 × `partial_credit` to its reward
  (`rlvr_speedrun/grpo.py`). An expression that uses exactly the puzzle's numbers earns 0.5 plus up to
  0.5 for closeness to the target (capped at 0.99). One with two or more numbers but the wrong
  multiset earns 0.25 × their overlap. A bare number, a malformed answer or a correct one earns no
  partial credit.
- `description`.

Everything else is D 004 (and so D 002):

- 112 min of pretraining from scratch on the pinned FineWeb-Edu shards, plus published DeepMind
  mathematics_dataset arithmetic (question-answer pairs, as published). See D 002's README for every
  external dataset with its revision.
- Then 234 min (14,040 s) of fp32 GRPO, with zero-advantage rows skipped.
- Sampler: `constrain: true`, T = 1, no think budget.

The gating screen was `calibration/research_r33/d004rl_shape05_lr6e-6_h200`, run on the H200 in the
5-seed form. Its in-loop final dev scores were 0.205 / 0.183 / 0.212 / 0.215 / 0.190 (mean 0.201),
against D 004's dev mean of 0.159 (`docs/research/LOG.md`, 2026-11-06 12:40).

The seeds ran from 2026-11-06 20:39 to 2026-11-07 16:43 EST on mit_preemptable H200 SXM workers
(`NVIDIA H200`) on MIT Engaging. Each seed ran `scripts/research/run_item.sh`: first
`scripts/warm_start.sh`, untimed, then `STAGE=all scripts/run_seed_v3.sh`. The software is this
repository at the commits listed in `docs/research/LOG.md`, with `uv.lock`'s package versions. No
code differs from the repository.

| seed | node | test ladder | pass@16 | in-loop final dev | pretrain_s | total_s | within 21,600 s | warm start |
|---|---|---|---|---|---|---|---|---|
| 0 | node4300 | 0.2150 | 0.2700 | 0.193 | 6731 | 20915 | yes | 53 s |
| 1 | node4300 | 0.1711 | 0.2408 | 0.187 | 6732 | 20876 | yes | 10 s |
| 2 | node4802 | 0.2183 | 0.2892 | 0.210 | 6728 | 20875 | yes | 13 s |
| 3 | node4300 | 0.2516 | 0.3125 | 0.217 | 6729 | 20874 | yes | 6 s |
| 4 | node4802 | 0.2191 | 0.2742 | 0.200 | 6729 | 20883 | yes | 6 s |

- Test ladder score, mean of all five seeds: 0.2150 ± 0.0287. The in-loop final dev mean is
  0.201, and the mean pass@16 is 0.2773.
- On the clock, the seeds took 347.9-348.6 min of the 360.
- `warm_start.log` lists every warm start.
- No attempt was preempted, so there is no `extra/`. No seed was dropped or rerun.
