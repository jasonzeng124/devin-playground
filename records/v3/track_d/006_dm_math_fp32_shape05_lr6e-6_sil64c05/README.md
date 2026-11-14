# Track D 006: D 005 with self-imitation

D 005's recipe (`records/v3/track_d/005_dm_math_fp32_shape05_lr6e-6`) plus **self-imitation**: during RL, the
policy's own verifier-correct completions from this run go into a buffer, and each step replays up to 64 of them
as extra rows. Pretraining is unchanged. Mean test ladder score of all five seeds: **0.2444 ± 0.0158**, against
D 005's 0.2150 ± 0.0287. Every seed finished within the 6-h budget.

## What ran

`config.json` is D 005's config with these changes:

- `sil_rows: 64`, `sil_coef: 0.5`, `sil_buffer: 4096`, `sil_by_level: true` (self-imitation; `rlvr_speedrun/grpo.py`,
  off by default).
- `description`.

What self-imitation does, and its limits (the maintainer allowed it on these terms, `docs/research/INBOX.md`
2026-11-10 14:50 and 2026-11-12 11:56):

- Every completion the verifier accepts during the run's own GRPO rollouts goes into a buffer, token for token as
  sampled, together with the generator puzzle it answered. The verifier is the only filter.
- The buffer is in memory only. It starts empty at the start of every seed's timed run and is never saved or
  carried between runs or seeds.
- Each optimizer step replays up to 64 buffer entries (never one entry twice in a step) as extra rows of the same
  step, each weighing like a policy-gradient row with advantage 0.5: a plain log-likelihood term, with no entropy
  bonus, KL or importance weight. Replayed completions keep their original puzzle.
- The buffer keeps the newest 4,096 entries per puzzle level, and the 64 draws are spread evenly over the levels
  that hold anything.
- Nothing else enters the buffer: no other model, no solver, no generated or relabelled puzzles.

Everything else is D 005 (and so D 004 and D 002):

- 112 min of pretraining from scratch on the pinned FineWeb-Edu shards, plus published DeepMind
  mathematics_dataset arithmetic (question-answer pairs, as published). See D 002's README for every
  external dataset with its revision.
- Then 234 min (14,040 s) of fp32 GRPO at constant lr 6e-6 with partial credit 0.5 (`shaping`), with
  zero-advantage rows skipped.
- Sampler: `constrain: true`, T = 1, no think budget.

The gating screen was `calibration/research_r61/d005rl_sil64c05`: D 005's RL phase with self-imitation, run on the
H200 from D 005's own pretrained checkpoints, in the 3-seed form. Its in-loop final dev scores were 0.250 / 0.247 /
0.237 against D's line 0.231 (D 005's dev mean 0.201 + 0.03), paired with D 005's seeds +0.057 / +0.060 / +0.027
(`docs/research/LOG.md`, 2026-11-12 20:0x). A fourth screen seed, run afterwards, ended at 0.228. The earlier
self-imitation screen on D 004's checkpoints was r59 (LOG 2026-11-12 06:08).

The seeds ran from 2026-11-13 06:14 to 20:27 EST on mit_preemptable H200 SXM workers (`NVIDIA H200`) on MIT
Engaging. Each seed ran `scripts/research/run_item.sh`: first `scripts/warm_start.sh`, untimed, then
`STAGE=all scripts/run_seed_v3.sh`. The software is this repository at the commits listed in
`docs/research/LOG.md`, with `uv.lock`'s package versions. No code differs from the repository.

| seed | node | test ladder | pass@16 | in-loop final dev | pretrain_s | total_s | within 21,600 s | warm start |
|---|---|---|---|---|---|---|---|---|
| 0 | node4601 | 0.2252 | 0.2883 | 0.220 | 6739 | 20932 | yes | 49 s |
| 1 | node4801 | 0.2487 | 0.3304 | 0.238 | 6730 | 20894 | yes | 41 s |
| 2 | node4801 | 0.2662 | 0.3467 | 0.243 | 6729 | 20881 | yes | 9 s |
| 3 | node4603 | 0.2335 | 0.3233 | 0.220 | 6733 | 20931 | yes | 42 s |
| 4 | node4801 | 0.2485 | 0.3375 | 0.225 | 6729 | 20892 | yes | 6 s |

- Test ladder score, mean of all five seeds: 0.2444 ± 0.0158 (D 005: 0.2150 ± 0.0287). Paired by seed with
  D 005 the differences are +0.0102 / +0.0776 / +0.0479 / -0.0181 / +0.0294. The in-loop final dev mean is
  0.229, and the mean pass@16 is 0.3253.
- On the clock, the seeds took 348.0-348.9 min of the 360.
- `warm_start.log` lists every warm start.
- No attempt was preempted, so there is no `extra/`. No seed was dropped or rerun.
