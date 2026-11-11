# Status

(maintained by the research agent; see AGENT.md)

Updated 2026-11-10 19:31 EST. The run ends 2026-12-19 (INBOX 11-10 10:00). Since 11-10 the agent runs on
Fable 5.2. The handover entry (holders, open arms, queues, the plan for the extra month) is LOG
2026-11-10 10:33.

INBOX 11-10 14:50 and 14:54: self-imitation on the policy's own verified samples is allowed within
limits (PLAN D11). If an arm built on it meets the promotion rule, that is flagged here, at the
top, before the record starts. It is coded and tested (`sil_rows`, off by default), and its first
arm, r59, is first in the track D B200 queue (LOG 16:16 EST). One unranked screen of hindsight
relabelling is allowed (PLAN D12); its code is next.

INBOX 11-10 14:58 and 15:04: the maintainer's own ideas are judged in `~/rlvr-ideas/STATUS.md`
(private, outside this checkout; the first pass is done). RULES v3.4 lets a run choose among its
own checkpoints or roll back, on the clock and never from the test ladder (PLAN A8). On A 005's
logs, keeping the best in-loop dev checkpoint gives dev 0.364 (SD 0.027) against the final 0.313
(SD 0.087). The option (`keep_best_dev`) is coded and tested, and r58, r54's config with it on,
is now A 006's promotion screen (LOG 15:46 EST).

## Records (H200 board; rules v3.4 since 11-10, no record so far chose a checkpoint)

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

## Jobs (8 GPU jobs: 1 running, 7 pending; at most 6 can run together)

The first B200 worker started at 15:03 EST. The H200 lines are days long (LOG 2026-11-10 14:59 EST), so
step-capped screens moved to B200, where the line is short.

| job | partition | GPU | queue | state |
|---|---|---|---|---|
| 28623801 | mit_normal_gpu | B200 | `screens_d_b200`, 3 runs packed | **running** since 15:03 EST on node5600-c1: r56 seeds 0-2; its 6-h limit ends 21:03 EST |
| 28646346 | mit_normal_gpu | B200 | `screens_b200`, 1 run | pending since 19:29 EST; r60 seed 0 is first in its queue |
| 28623803 | preemptable | B200 | `screens_d_b200`, 3 runs packed | pending since 14:53 EST |
| 28473022 | mit_normal_gpu | H200 | `h200` | pending (Slurm's estimate 11-14 05:30) |
| 28595679 | mit_normal_gpu | H200 | `screens_a_h200` | pending (no estimate now) |
| 28595678 | preemptable | H200 | `h200` | pending |
| 28316532 | preemptable | H200 | `h200` | preempted 13:36:50 EST, requeued, pending; holds no item |
| 28486959 | preemptable | H200 | `screens_a_h200` | preempted 11:00 EST, requeued, pending; holds no item |

The cap: `mit_normal_gpu` runs at most 2 GPUs per user, so of the four jobs there two run at a time,
plus the four on preemptable. While the B200 normal jobs run, the H200 normal jobs wait (they keep
their age). 28523219 was cancelled at 14:52 EST. 28623802 ended at 18:43 EST when r51 seed 0
finished (less than 200 min were left).

Not mine, left alone: the `eng-board1` array, 28597816, 28597977, 28600134 (`tc`), 28623041 (`v12`).

## Queues

- `screens_d_b200` (track D, B200, 10 runs waiting, about 22 GPU-hours packed 3 per GPU at r56's
  1.56 s per step): r59 seeds 0-2, r57 seeds 0-2, r56 seeds 3-4, r57 seeds 3-4. r56 seeds 0-2 are
  running.
- `screens_b200` (track A, B200, 7 runs, about 28 GPU-hours, one run per 6-h job): r60 seed 0, r51
  seed 1, r60 seed 1, r51 seed 2, r60 seed 2, r51 seeds 3-4.
- `h200` (H200 only, about 31 GPU-hours): r58 seeds 0-1, r55 seeds 0-1, r53 seeds 0-2.
- `screens_a_h200` (H200 only, about 6 GPU-hours): r58 seed 2, r55 seed 2.

## Open arms

| arm | track | what | so far | decision |
|---|---|---|---|---|
| r58 `a005_ent05_keepbest_h200` | A | A 005 + entropy floor (`entropy_coef` 0.05 below `entropy_target` 0.1) + `keep_best_dev` (the run ends on its best in-loop dev eval's weights, RULES v3.4); the H200 promotion screen for A 006. It took r54's place before r54 ran | not started; first on both H200 queues | the kept dev reads about 0.01-0.02 high (best of eight evals), so it is read against 0.305: seeds 0-2 all at 0.305 or above and none collapsed start record A 006 with the option on (5 fixed seeds). Otherwise seeds 3-4 and the five-seed form. The last eval is reported beside the kept one |
| r51 `a005_os2` | A | A 005 + oversample 2, 400 steps on the 500-step schedule, now on B200 | seed 0 (LOG 19:31 EST): 0.387 and 0.390 at steps 300 and 350, then 0.053 at step 400. Ahead of A 005's seed 0 per step (+0.03 to +0.11), not per unit of time (1.8x per step), and no higher than r40's ceiling. Seeds 1-4 queued | seeds 1-2 say whether the ceiling or the collapse repeats; A7 (b) (oversampling with the floor) waits for them |
| r60 `a005_ent05_cos875_s700_keepbest` | A | r58's config on a longer schedule (PLAN A7 (a)): cosine over 875 steps, stopped at 700, step-capped on B200, the 17,400-s budget still on | seed 0 first in `screens_b200` | evals past step 400 against r40's 0.372; a kept dev of 0.41 or above on two of three seeds keeps the schedule in the 11-18 box's track A batch and leads to an H200 gate; otherwise that half of the box is planned again by 11-16 |
| r56 `d004rl_shape05_lr6e-6_mini3_s15k` | D | r30 + the mini levels M2, M3, M6 in the training mix (PLAN D6), 15,000 steps, B200 packed | seeds 0-2 running since 15:03 EST at 1.56 s per step (15,000 steps take about 6.5 h, so each seed continues in a second job); seeds 3-4 queued | paired with r30 (0.192 / 0.180 / 0.220 / 0.233 / 0.193). A clear gain: H200 gate against 0.231. Clearly negative: drop r55 |
| r59 `d004rl_shape05_lr6e-6_sil64c05_s15k` | D | r30 + self-imitation (PLAN D11): each step replays up to 64 of the run's earlier verifier-correct samples, drawn evenly over levels, each weighing like a policy-gradient row with advantage 0.5; 15,000 steps, B200 packed | seeds 0-2 first in `screens_d_b200` (LOG 16:16 EST) | paired with r30. Read `entropy`, `informative_groups` and `t_update` against r30's. A clear gain: seeds 3-4, a sweep on the 11-18 box, an H200 gate against 0.231 |
| r57 `d004rl_shape05_lr6e-6_noL5_s15k` | D | r30 with no L5 prompts in training (PLAN D8), 15,000 steps, B200 packed | queued | paired with r30 |
| r55 `d005rl_mini3` | D | RL from D 005's checkpoints with the mini levels, time-budgeted on H200 | queued in both H200 queues after r54 | paired with D 005's seeds 0-2 (0.193 / 0.187 / 0.210) |
| r53 `d005_w768_l24` | D | depth 24 at width 768 | last in `h200` | three seeds, then decide on seeds 3-4 |

Closed 11-10: r52 (width 1280), +0.004 paired with D 005 over four seeds; seed 4 was preempted
and is not rerun. Width stays out of the stack.

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

- When a B200 worker starts: check its first minutes for quick failures (requeue a FAILED item
  once). 28623801 passed that check, and r56's training mix holds the mini levels (LOG 15:24 EST).
  28623801 ends at 21:03 EST at the latest.
- A `mit_normal_gpu` worker that ends at its 6-h limit is not requeued: put its
  `queue/running/<job>-k` items back at the front of their queue and submit a replacement, keeping
  at most 6 GPUs able to run together.
- Code, each option off by default and tested. Done: keeping the best in-loop dev checkpoint
  (`keep_best_dev`, with the validator check) and self-imitation (`sil_rows`). Next, in this
  order: the unranked relabelling screen (D12), rollback and train on (A8 (c), needed by 11-16 for
  the box).
- r59 (self-imitation): when its seeds start, check the first minutes (the replay runs on a GPU for
  the first time) and that `sil_rows` reaches 64 within the first few hundred steps.
- `~/rlvr-ideas/STATUS.md`: update it whenever a verdict or a result changes.
- r58 seeds 0-2 (about 2.5 h each on H200): apply the rule in its row; on a pass, create and commit
  the A 006 record config (with `keep_best_dev`; the README says the run chose a checkpoint) and
  queue its 5 seeds on H200 record workers.
- r60 and r51 (about 4 h per run on B200, one per 6-h job): when 28646346 starts, check its first
  minutes; each time its job ends, submit the next `screens_b200` worker.
- r56 and r57 (4-6 h per run packed): log each paired with r30.
- The 11-18 box: queue files, configs and the copy list committed by 11-16.

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
