# Status

(maintained by the research agent; see AGENT.md)

Updated 2026-11-10 21:46 EST. The run ends 2026-12-19 (INBOX 11-10 10:00). Since 11-10 the agent runs on
Fable 5.2. The handover entry (holders, open arms, queues, the plan for the extra month) is LOG
2026-11-10 10:33.

INBOX 11-10 14:50 and 14:54: self-imitation on the policy's own verified samples is allowed within
limits (PLAN D11). If an arm built on it meets the promotion rule, that is flagged here, at the
top, before the record starts. It is coded and tested (`sil_rows`, off by default), and its first
arm, r59, is first in the track D B200 queue (LOG 16:16 EST). One unranked screen of hindsight
relabelling is allowed (PLAN D12). It is coded and tested too (`hindsight_rows`, off by default,
refused in records by the validator; LOG 21:46 EST EST), and its two arms run on the 11-18 box.

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

## Jobs (9 GPU jobs: 0 running, 9 pending; at most 6 can run together)

Nothing of mine has run since 20:54 EST, when 28623801 completed with r56 seeds 0-2 done. Every
pending job waits on priority: mine is about 365,000 on `mit_normal_gpu` (low fairshare after three
weeks of use) against 515,000-565,000 for the jobs planned onto the free GPUs (LOG 21:21 EST).

| job | partition | GPU | queue | state |
|---|---|---|---|---|
| 28653623 | mit_normal_gpu | B200 | `screens_d_b200`, 3 runs packed | pending since 21:14 EST; r59 seeds 0-2 are first in its queue |
| 28646346 | mit_normal_gpu | B200 | `screens_b200`, 1 run | pending since 19:29 EST (Slurm's estimate 11-11 02:38 EST); r60 seed 0 is first in its queue |
| 28653733 | mit_normal_gpu | H200 | `gate_a_h200` | pending since 21:18 EST; `--time-min=03:15:00` and `RESERVE_MIN=180`, so it can start in a 3.25-h gap; r58 seeds 1-2 only |
| 28473022 | mit_normal_gpu | H200 | `h200` | pending (Slurm's estimate 11-14 15:00 EST); r58 seed 0 is first in its queue |
| 28595679 | mit_normal_gpu | H200 | `screens_a_h200` | pending (Slurm's estimate 11-15 12:30 EST); r55 seed 2 |
| 28623803 | preemptable | B200 | `screens_d_b200`, 3 runs packed | pending since 14:53 EST |
| 28595678 | preemptable | H200 | `h200` | pending |
| 28316532 | preemptable | H200 | `h200` | preempted 13:36:50 EST, requeued, pending; holds no item |
| 28486959 | preemptable | H200 | `screens_a_h200` | preempted 11:00 EST, requeued, pending; holds no item |

The cap: `mit_normal_gpu` runs at most 2 GPUs per user, so of the five jobs there two run at a time,
plus the four on preemptable. Slurm's estimates move by days between readings; squeue prints them
in local time (EST).

Not mine, left alone: the `eng-board1` array, 28597816, 28597977, 28600134 (`tc`), 28623041 (`v12`).

## Queues

- `screens_d_b200` (track D, B200, 10 runs, about 19 GPU-hours packed 3 per GPU; r56's 15,000 steps
  took 5.6-5.8 h, so three runs fit one 6-h job): r59 seeds 0-2, r57 seeds 0-2, r56 seeds 3-4, r57
  seeds 3-4. r59's replay adds to each update, so its seeds may be cut at the 6-h limit and finish
  in a second job.
- `screens_b200` (track A, B200, 7 runs, about 28 GPU-hours, one run per 6-h job): r60 seed 0, r51
  seed 1, r60 seed 1, r51 seed 2, r60 seed 2, r51 seeds 3-4.
- `gate_a_h200` (H200 only, about 5 GPU-hours, read by the short-gap worker 28653733): r58 seeds 1-2.
  Nothing longer than an r58 seed goes in this queue.
- `h200` (H200 only, about 29 GPU-hours): r58 seed 0, r55 seeds 0-1, r53 seeds 0-2.
- `screens_a_h200` (H200 only, about 4 GPU-hours): r55 seed 2.

## Open arms

| arm | track | what | so far | decision |
|---|---|---|---|---|
| r58 `a005_ent05_keepbest_h200` | A | A 005 + entropy floor (`entropy_coef` 0.05 below `entropy_target` 0.1) + `keep_best_dev` (the run ends on its best in-loop dev eval's weights, RULES v3.4); the H200 promotion screen for A 006. It took r54's place before r54 ran | not started; seed 0 is first in `h200`, seeds 1-2 are in `gate_a_h200` (a worker that can start in a 3.25-h gap) | the kept dev reads about 0.01-0.02 high (best of eight evals), so it is read against 0.305: seeds 0-2 all at 0.305 or above and none collapsed start record A 006 with the option on (5 fixed seeds). Otherwise seeds 3-4 and the five-seed form. The last eval is reported beside the kept one |
| r51 `a005_os2` | A | A 005 + oversample 2, 400 steps on the 500-step schedule, now on B200 | seed 0 (LOG 19:31 EST): 0.387 and 0.390 at steps 300 and 350, then 0.053 at step 400. Ahead of A 005's seed 0 per step (+0.03 to +0.11), not per unit of time (1.8x per step), and no higher than r40's ceiling. Seeds 1-4 queued | seeds 1-2 say whether the ceiling or the collapse repeats; A7 (b) (oversampling with the floor) waits for them |
| r60 `a005_ent05_cos875_s700_keepbest` | A | r58's config on a longer schedule (PLAN A7 (a)): cosine over 875 steps, stopped at 700, step-capped on B200, the 17,400-s budget still on | seed 0 first in `screens_b200` | evals past step 400 against r40's 0.372; a kept dev of 0.41 or above on two of three seeds keeps the schedule in the 11-18 box's track A batch and leads to an H200 gate; otherwise that half of the box is planned again by 11-16 |
| r56 `d004rl_shape05_lr6e-6_mini3_s15k` | D | r30 + the mini levels M2, M3, M6 in the training mix (PLAN D6), 15,000 steps, B200 packed | seeds 0-2 (LOG 21:21 EST): 0.217 / 0.212 / 0.222 against r30's 0.192 / 0.180 / 0.220, paired +0.019 (p 0.17), ahead at every eval from step 8,000. All of the gain is at L2 (0.19 against 0.12). Seeds 3-4 queued | a small gain, short of the D line alone (about 0.22 against 0.231): r55 stays, the mini levels go into the D10 stack, and D6's follow-ups for the box (M2 alone, a larger share of it, intermediate levels for L3-L5) are fixed by 11-16 |
| r59 `d004rl_shape05_lr6e-6_sil64c05_s15k` | D | r30 + self-imitation (PLAN D11): each step replays up to 64 of the run's earlier verifier-correct samples, drawn evenly over levels, each weighing like a policy-gradient row with advantage 0.5; 15,000 steps, B200 packed | seeds 0-2 first in `screens_d_b200` (LOG 16:16 EST) | paired with r30. Read `entropy`, `informative_groups` and `t_update` against r30's. A clear gain: seeds 3-4, a sweep on the 11-18 box, an H200 gate against 0.231 |
| r57 `d004rl_shape05_lr6e-6_noL5_s15k` | D | r30 with no L5 prompts in training (PLAN D8), 15,000 steps, B200 packed | queued | paired with r30 |
| r55 `d005rl_mini3` | D | RL from D 005's checkpoints with the mini levels, time-budgeted on H200 | seeds 0-1 in `h200` after r58 seed 0; seed 2 alone in `screens_a_h200` | paired with D 005's seeds 0-2 (0.193 / 0.187 / 0.210) |
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

- Nothing runs now. When a worker starts: check its first minutes for quick failures (requeue a
  FAILED item once). When 28653733 (the short-gap H200 worker) ends, look at `gate_a_h200`; if
  r58 seeds are left there, submit another worker like it (LOG 21:21 EST has the line).
- A `mit_normal_gpu` worker that ends at its 6-h limit is not requeued: put its
  `queue/running/<job>-k` items back at the front of their queue and submit a replacement, keeping
  at most 6 GPUs able to run together.
- Code, each option off by default and tested. Done: keeping the best in-loop dev checkpoint
  (`keep_best_dev`, with the validator check), self-imitation (`sil_rows`) and hindsight relabelling
  for the unranked screen (`hindsight_rows`, D12). Next: rollback and train on (A8 (c), needed by
  11-16 for the box).
- r59 (self-imitation): when its seeds start, check the first minutes (the replay runs on a GPU for
  the first time) and that `sil_rows` reaches 64 within the first few hundred steps.
- `~/rlvr-ideas/STATUS.md`: update it whenever a verdict or a result changes.
- r58 seeds 0-2 (about 2.5 h each on H200): apply the rule in its row; on a pass, create and commit
  the A 006 record config (with `keep_best_dev`; the README says the run chose a checkpoint) and
  queue its 5 seeds on H200 record workers.
- r60 and r51 (about 4 h per run on B200, one per 6-h job): when 28646346 starts, check its first
  minutes; each time its job ends, submit the next `screens_b200` worker.
- r59, r57 and r56 seeds 3-4 (5.6-5.8 h per run packed, r59 longer): log each paired with r30.
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
