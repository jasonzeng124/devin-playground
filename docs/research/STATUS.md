# Status

(maintained by the research agent; see AGENT.md)

Updated 2026-11-10 10:33 EST. The run ends 2026-12-19 (INBOX 11-10 10:00). Since 11-10 the agent runs on
Fable 5.2. The handover entry (holders, open arms, queues, the plan for the extra month) is LOG
2026-11-10 10:33.

## Records (H200 board, rules v3.3)

| track | record | test, 5 seeds | in-loop dev mean | state |
|---|---|---|---|---|
| A | `records/v3/track_a/004_ladder_constrained_h200` | 0.2671 ± 0.0565 | 0.255 | **holder** |
| A | `records/v3/track_a/005_fp32_mal0_cos500_s400` | 0.3400 ± 0.0815 | 0.313 | valid, not a record (Welch vs A 004 p = 0.072) |
| A | `records/v3/track_a/002_fp32_skip_cosine` | 0.2784 ± 0.0904 | | valid, short of A 004 (p = 0.41) |
| D | `records/v3/track_d/005_dm_math_fp32_shape05_lr6e-6` | 0.2150 ± 0.0287 | 0.201 | **holder** (Welch vs D 004 p = 0.0034) |
| D | `records/v3/track_d/004_dm_math_fp32_h200_warm` | 0.1552 ± 0.0217 | 0.159 | held before D 005 |
| D | `records/v3/track_d/003_dm_math_fp32_h200` | | | failed: seeds 0 and 2 over budget |

The H100 board closed on 10-24 (A 001 0.2481; D 002 0.1512 ± 0.0202).

Promotion lines (the holder's dev mean + 0.03): **A 0.285, D 0.231**. Three-seed form: every seed
at or above the line, none collapsed. Five-seed form: mean at or above the line, at least 4 of 5
seeds above the holder's dev mean (A 0.255, D 0.201), at most one collapsed. The gating screen
runs on the H200. A 006 needs a test mean of about 0.33 to pass Welch against A 004 (LOG 11-10).

## Jobs (6 of 6 GPUs)

| job | partition | queue | state |
|---|---|---|---|
| 28316532 | preemptable | `h200` | running r52 seed 3 on node4300 since 05:01 EST; ends about 11:00 EST |
| 28486959 | preemptable | `screens_a_h200` | running r51 seed 0 on node4301 since 09:48 EST; ends about 13:45 EST |
| 28473022 | mit_normal_gpu | `h200` | pending |
| 28595678 | preemptable | `h200` | pending |
| 28523219 | preemptable | `screens_a_h200` | pending |
| 28595679 | mit_normal_gpu | `screens_a_h200` | pending |

Not mine, left alone: the `eng-board1` array, 28597816, 28597977.

## Queues

- `h200` (track D, about 36 GPU-hours): r52 seed 4; r55 seed 0; r53 seed 0; r55 seeds 1-2; r53
  seeds 1-2.
- `screens_a_h200` (track A, about 23 GPU-hours): r54 seeds 0-2; r51 seeds 1-4.

## Open arms

| arm | track | what | so far | decision |
|---|---|---|---|---|
| r54 `a005_ent05_h200` | A | A 005 + entropy floor (`entropy_coef` 0.05 below `entropy_target` 0.1); the H200 promotion screen for A 006 | queued | seeds 0-2 all at 0.285 or above and none collapsed: start record A 006 (5 fixed seeds). Otherwise seeds 3-4 and the five-seed form |
| r51 `a005_os2` | A | A 005 + oversample 2 | seed 0 running | compare with A 005's in-loop dev 0.313 |
| r55 `d005rl_mini3` | D | RL from D 005's checkpoints with the 3-number mini levels M2, M3, M6 in the training mix | queued | paired with D 005's seeds 0-2 (0.193 / 0.187 / 0.210) |
| r52 `d005_w1280_l12` | D | width 1280 (276M) | seeds 0-2 mean 0.208, +0.011 paired; seed 3 running, seed 4 queued | close on five seeds |
| r53 `d005_w768_l24` | D | depth 24 at width 768 | queued | three seeds, then decide on seeds 3-4 |

## AWS boxes (INBOX 11-10 10:40: back on, screens only, about 50/50 A/D)

| block (EST) | GPUs | ssh | launch template |
|---|---|---|---|
| 11-18 06:30 to 11-19 06:30 | 8x H100 | zjason@174.129.98.131 | `rlvr-box-cr-0261a6f8bb30fc123` |
| 11-28 06:30 to 11-29 06:30 | 8x H200 (the maintainer may take it) | zjason@100.29.118.132 | `rlvr-box-cr-08927241900b5884c` |

- No scheduled launch. From the block start: `aws ec2 describe-instances --region us-east-1`, then
  `aws ec2 run-instances --region us-east-1 --launch-template LaunchTemplateName=<template>`, retried
  every couple of minutes. No other AWS call; the $500 budget stays suspended.
- Engaging maintenance is 11-17 00:00-21:00 EST. The 11-18 queue files, configs and copy list are
  committed by 11-16.
- While a box is up: wake every 20-30 min, `box_sync.sh from` every wake and a last time before
  06:00 EST on the end day, and bring resume state back (INBOX 11-04).
- The plan for each box is in PLAN ("Box days") and LOG 11-10 10:33.

## Next

- r52 seed 3 ends about 11:00 EST: log it with seeds 0-2. The pending workers then pop r52 seed 4,
  r55 and r53.
- r54 seeds 0-2 need about 2.5 h each once the track A workers start (A 005's seeds took 2.1-2.5 h):
  apply the rule above.
- r55 seed 0: check the first train log for M2/M3/M6 puzzles in the mix.
- The D7 sampling profile, at a later full wake.
- QUESTIONS 11-10 (waiting for INBOX): hindsight relabelling (assumed not allowed), self-imitation
  on verified samples (not started), mini levels (for information).

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q` (two-rank tests fail on the login node otherwise).
- node4900 is excluded from every worker (CPUs oversubscribed). node2500 is excluded from every H200
  worker (track A ran about 3x slow there).
- Workers exit on empty queues: keep each queue stocked ahead of every worker's next pop.
- Items with `no_save` and a pretraining stage (r50, r52, r53) log FAILED in `worker.log` after their
  results are written (`mv` of a missing `final`). The results are valid.
- Record seeds: check `clock.json` `within_budget` and `warm_start.log` as soon as a seed finishes.
- Step-capped runs (`budget_steps`) log `clock_s` as the step clock (500 steps = 17,400 s).
- Track D pretraining is time-budgeted and its speed varies by node; read D results with their
  tokens/s.
- B200 workers: start them a few minutes apart (the first Triton compiles race on the shared cache).
- LOG and STATUS times come from `date`, in EST (`worker.log` is UTC).
