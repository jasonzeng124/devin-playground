# Status

(maintained by the research agent; see AGENT.md)

Updated 2026-11-16 18:47 EST. The run ends 2026-12-19. The agent runs on Opus 5.5.

**D 007 promotion (13:54 EST 11-16), built on self-imitation, so said here before its seeds start (INBOX 11-12).**
Arm r81, `calibration/research_r81/d006_batched_muon56_rl17400`: D 006 (which already uses self-imitation, same limits)
plus `constrain_batched`, with Muon pretraining for 56 min instead of AdamW for 112 and the 3,360 s moved to RL. H200
final dev 0.262 / 0.268 / 0.275 (mean 0.268), each at or above D's line 0.259: the 3-seed form. Record config
`records/v3/track_d/007_dm_math_sil64c05_batched_muon56` is r81's minus `no_save`. Seeds 0-3 run on preemptable H200
workers sized to end before the 11-17 maintenance; seed 4 after it. Risk: r81 seed 2's process took about 15 min to exit
after writing its result (GRPO wall 18,430 s for 17,428 s of loop), which put its clock at 21,770 s against 21,600.
The other 15 recent D runs had 55-220 s of overhead. See QUESTIONS 11-16 13:54.

**Record A 007 is the new track A holder (2026-11-16 13:54 EST).** Test 0.4527 ± 0.0113 over five seeds (0.4557 /
0.4391 / 0.4527 / 0.4693 / 0.4464, from `confirm_v3.json`) against A 006's 0.4123 ± 0.0127: one-sided Welch
p = 0.0004. A 006 with 96 prompts per step, the cosine over 583 steps, stopped at step 467; kept dev mean 0.429, so
A's line is now 0.459. Seed 1 was preempted twice (partials in `extra/`).

**Record D 006 is the track D holder (2026-11-13).** Test 0.2444 ± 0.0158; dev mean 0.229; D's line 0.259.

**Jobs (13:54 EST 11-16).** All earlier workers have ended. A 007 is done. r81 seeds 0-2 are done (closed by
promotion). r87 (B200, seed 0, not gated): muon1338m_s14180 final dev 0.240 against d004_ctl_s11441 0.263, one paired
seed. Starting now: D 007 seeds 0-3 on four preemptable H200 workers.
r88 (U18 answer-opening mask; B200, not gated): final dev 0.235 (seed 0) and 0.237 (seed 1); seed 0 is
-0.028 against its control. r87's control seed 1 runs on B200 (29387381) until about 21:20 EST, for the seed 1 pair.
**Next decisions:** D 007 seed by seed; seed 4 after the maintenance (11-17 21:00 EST); r88 seed 1 against r87's control seed 1
(tonight); the 11-18 H100 box (`box_a_1118`: r85, A 007 arms; `box_d_1118`: r86, Muon
checkpoints with more RL steps).

**B200 first (INBOX 11-13 23:43 and 23:45).** Progress comes before records. Results on B200 are marked "not yet gated"
until an H200 gate runs. INBOX 11-14 22:03: a gate or record may take an H200 slot when the agent judges that best.

INBOX notes in force: self-imitation is allowed within the 11-12 limits (empty buffer per seed, never carried between
runs, completions replayed verbatim with their own puzzle, the verifier the only filter), and records that use it say so
in their README. Hindsight relabelling stays out of records and promotion screens. The maintainer's ideas are judged in
`~/rlvr-ideas/STATUS.md` (private, outside this checkout). RULES v3.5: a record may train and score outside vLLM.

## Records (H200 board; rules v3.5)

| track | record | test, 5 seeds | in-loop dev mean | state |
|---|---|---|---|---|
| A | `records/v3/track_a/007_fp32_mal0_p96_cos583_s467_ent05_keepbest` | 0.4527 ± 0.0113 | 0.429 (kept) | **holder** (Welch vs A 006 p = 0.0004) |
| A | `records/v3/track_a/006_fp32_mal0_cos500_s400_ent05_keepbest` | 0.4123 ± 0.0127 | 0.388 (kept) | held before A 007 (Welch vs A 004 p = 0.0019) |
| A | `records/v3/track_a/004_ladder_constrained_h200` | 0.2671 ± 0.0565 | 0.255 | held before A 006 |
| A | `records/v3/track_a/005_fp32_mal0_cos500_s400` | 0.3400 ± 0.0815 | 0.313 | valid, not a record (Welch vs A 004 p = 0.072) |
| A | `records/v3/track_a/002_fp32_skip_cosine` | 0.2784 ± 0.0904 | | valid, short of A 004 (p = 0.41) |
| D | `records/v3/track_d/007_dm_math_sil64c05_batched_muon56` | | | running (promoted from r81 on 11-16) |
| D | `records/v3/track_d/006_dm_math_fp32_shape05_lr6e-6_sil64c05` | 0.2444 ± 0.0158 | 0.229 | **holder** (Welch vs D 005 p = 0.045; self-imitation) |
| D | `records/v3/track_d/005_dm_math_fp32_shape05_lr6e-6` | 0.2150 ± 0.0287 | 0.201 | held before D 006 (Welch vs D 004 p = 0.0034) |
| D | `records/v3/track_d/004_dm_math_fp32_h200_warm` | 0.1552 ± 0.0217 | 0.159 | held before D 005 |
| D | `records/v3/track_d/003_dm_math_fp32_h200` | | | failed: seeds 0 and 2 over budget |

The H100 board closed on 10-24 (A 001 0.2481; D 002 0.1512 ± 0.0202).

Promotion lines (the holder's dev mean + 0.03): **A 0.459 (kept dev, A 007's 0.429), D 0.259 (D 006's 0.229)**.
Three-seed form: every seed at or above the line, or +0.03 over its paired gate-screen seed, none collapsed. Five-seed
form: mean at or above the line, at least 4 of 5 seeds above the holder's dev mean, at most one collapsed. The gating
screen runs on the H200.

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

- D 007: check each seed's `clock.json` (`within_budget`) and `warm_start.log` as it finishes; a preempted seed's partial
  goes to `extra/` and the same seed reruns (the worker does this). After five seeds: README, validation against D 006,
  leaderboard.
- After the maintenance (11-17 21:00 EST): `box_sync.sh to` as a CPU job (docs/research/box_1118.md), then the 11-18 box
  from 06:30 EST with only the two allowed AWS commands; wake every 20-30 min while it is up.
- U18: r88 seed 0 is -0.028 against its control; read seed 1's pair when r87's control seed 1 ends (about 21:20 EST).
- `~/rlvr-ideas/STATUS.md`: update it whenever a verdict or a result changes.

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
