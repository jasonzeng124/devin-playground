# Status

(maintained by the research agent; see AGENT.md)

Updated 2026-11-18 01:19 EST. The run ends 2026-12-19. The agent runs on Opus 5.5.

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
seed. D 007 seeds 0-3 done (20:26 EST): test 0.261 / 0.254 / 0.265 / 0.273, each about 720 s inside the budget.
r88 (U18 answer-opening mask; B200, not gated): -0.028 and +0.020 against r87's control on seeds 0 and 1
(mean -0.004): no effect; U18 parked.
11-17 12:17 EST: the maintenance is over; D 007 seed 4 runs (29355448, node4802, from 16:00Z). r89 (U19 step 2, pretraining throughput: `--fused-loss`, micro-batch 32 and 64; B200, pretraining only, 56 min) on workers 29441392 and 29441400: +12 % and +16 % tokens per second at the same loss. r90 (speed-only benchmarks, 300 steps each, INBOX 11-17 12:25) on H200 worker 29445218. 11-17 13:35 EST: D 007 seed 4 was preempted at 13:19 after about 3 h 15 min; requeued (partial to `extra/`, seed 4 reruns). r89 final val 2.816 and 2.805 at 56 min against r82's 2.805: faster, not a better 56-min checkpoint, so the use is time moved to RL. r91 (r90's benchmarks on B200, worker 29448582): fused micro-batch 32 +14 % tok/s at matching losses. 14:07: r91 done: `--fused-loss --micro-batch 128` +21 % (B200). r92 (U19 step 3, H200, end to end, seeds 0-2, queue `screens_r92`, workers 29450693/94/702): r81 with that path, pretraining 46 min and RL 18,000 s, read paired against r81; plus a 46-min pretraining-only check on B200 (29450704). 15:12: that check matches r82's 56-min run step for step (final val 2.832 vs 2.805, within val noise), so 46 min on the fast path gives about the old 56-min checkpoint. INBOX 15:02: a speed change with the same maths is kept without a paired win; `--fused-loss --micro-batch 128` is the default D pretraining path for new arms. r93 (B200 worker 29456195, queue `screens_b200`): timing of `share_group_prompt` (new, off by default: the RL update runs each distinct prompt once), 300 steps from D 007 seed 0's start and final weights. 17:36: r94: the shared prompt with 65,536-token micro-batches cuts the update 14-18 % and fits the H200 (131,072 would not); that is the speed setting for new D arms. r95 (B200 worker 29468588): `time_update_parts` (new, off by default) times the update's parts (INBOX 17:14). INBOX 17:23: the 6-GPU cap counts running GPUs only; my H200 jobs are all still pending. 18:12: r95: the update is GPU-bound (head, optimizer negligible; Python grammar work 7-8 % of the step); sampling is 50-55 % of the step. r96 (B200) splits sampling's time. 20:32: r96: vLLM `generate` is 95 % of sampling (0.59 s of 0.62 s per step at the start, 0.46 of 0.48 at the end); a profile of it runs on B200 (29497515). INBOX 20:23 (U20): more use out of the B200s; records stay on the H200. 23:09: r97 (B200, not gated; queue `screens_d_b200`, preemptable workers 29497701/02/05): D 007's arm with the whole speed stack (46-min fused Muon pretraining, shared prompt, 65,536-token micro-batches) against D 007's arm, seeds 0-2, end to end. The r92 H200 workers and r90's bench are held (`scontrol hold`) to stay inside the cap; r97 contains r92's change. 11-18 01:18: r96's profile: vLLM's per-request Python bookkeeping, not the forward, is the sampling cost. A same-process sampler (`rollout_backend: "native"`, off by default, f9cf5c2) replaces it; r98 (B200 job 29508506) times it against vLLM.
RULES v3.6 (INBOX 11-16 22:35): a record seed over budget only from idle exit time counts as it is (QUESTIONS when it happens).
**Next decisions:** D 007 seed by seed; seed 4 after the maintenance (11-17 21:00 EST); the 11-18 H100 box (`box_a_1118`: r85, A 007 arms; `box_d_1118`: r86, Muon
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
| D | `records/v3/track_d/007_dm_math_sil64c05_batched_muon56` | 0.2629 (seeds 0-3) | 0.250 (seeds 0-3) | running: seed 4 rerunning after a preemption (11-17 13:19) |
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
