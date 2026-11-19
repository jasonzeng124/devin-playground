# Status

(maintained by the research agent; see AGENT.md)

Updated 2026-11-19 09:04 EST. The run ends 2026-12-19. The agent runs on Fable 5.2.

**The speed stack turns into score (11-19, LOG 11-19 02:05).** r101 (B200, RL only, one seed each, not yet gated): the
native arm ran 34,893 steps in 17,400 s and ended at dev 0.3033; the control ran 18,215 and ended at
0.2483 (its seed 1: 0.2383 after 20,032); level at equal steps. **r102 is its H200 promotion screen**: `calibration/research_r102/d007_speed_h200`
(D 007 end to end with the fused pretraining for 46 min and the native RL settings for 18,000 s), seeds 0-2 in
`queue/screens_r92.txt` on H200 preemptable workers 29450693/94/702 (released 01:52; still pending on `Priority`
at 09:04). Read against D's line 0.281 (D 007's dev mean 0.251 + 0.03); if the three-seed rule is met, start a
five-seed record attempt (r102's config without `no_save`). Watch the first run pass from pretraining into native
RL (never run end to end before).
On the 11-18 box the native sampler came out a little behind vLLM on dev at equal steps, by about 0.01 (finals
-0.021 over three finished pairs, SE 0.009; evaluation curves -0.008 over five pairs, SE 0.003; LOG 11-19 06:32),
against about twice the steps in the same time. Whether that gap comes from training or from the native greedy
evaluation is open; the test is in "Next".
INBOX 11-19 01:32 (critique of the box plan) is answered in the same LOG entry: speed stack first, fewer arms and
more seeds with the expected effect written beside the control's seed sd, the rest on U17/U19 and on settings the
doubled step count may have moved; each box queue is re-planned the day before (11-24, 11-27, 12-08).

**Record D 007 is the track D holder (2026-11-19 05:40 EST).** Test 0.2644 ± 0.0076 over five seeds (0.2606 /
0.2538 / 0.2646 / 0.2726 / 0.2703, from `confirm_v3.json`) against D 006's 0.2444 ± 0.0158: one-sided Welch
p = 0.0226. D 006 (self-imitation within the 11-12 limits, as its README says) with `constrain_batched`, Muon
pretraining for 56 min instead of AdamW for 112, and the 56 min moved to RL. In-loop dev mean 0.251, so D's line
is now 0.281. Every seed is within the budget (347.6-348.0 min of 360). Seed 4 was preempted once (partial in
`extra/`). Most of the gain is the extra RL steps that batched masking gives (16,318-19,735 against D 006's
10,068-11,380); the Muon split by itself is about level on dev (the record's README; r86).

**Record A 007 is the new track A holder (2026-11-16 13:54 EST).** Test 0.4527 ± 0.0113 over five seeds (0.4557 /
0.4391 / 0.4527 / 0.4693 / 0.4464, from `confirm_v3.json`) against A 006's 0.4123 ± 0.0127: one-sided Welch
p = 0.0004. A 006 with 96 prompts per step, the cosine over 583 steps, stopped at step 467; kept dev mean 0.429, so
A's line is now 0.459. Seed 1 was preempted twice (partials in `extra/`).

**Jobs (09:04 EST 11-19).** Two GPUs running, both B200s on node5602-c1 (`mit_normal_gpu`, queue
`screens_rl_b200`): 29618971 (started 08:41; ran r104, now r101's control seed 2) and 29626073 (started 08:48;
r101's native seed 1). Both items are traced and end at about 13:45 EST; neither worker has room for another
item. Pending: 29632479 (preemptable, same queue, 14 h, room for two items; waiting for nodes; r101's native seed
2 is the one item left in the queue); r102's three H200 preemptable workers (29450693/94/702, `Priority`). Held:
29497701/02/05 (B200 preemptable; r97/r99, superseded by r102) and 29445218 (H200; r90). If everything pending
starts that is 4 preemptable and 2 normal GPUs, the cap of six. **The B200 queue needs new items today.**

**Settled speed settings for new D arms (LOG 11-17 and 11-18; timed on B200; each is the same maths and off by
default).** Pretraining: `--fused-loss --micro-batch 128` (+21 % tokens per second; 46 min gives about the old
56-min checkpoint). RL: the shared prompt with 65,536-token micro-batches (update 14-18 % shorter; fits the H200),
the native sampler (`rollout_backend: "native"`, 1.40-1.60x faster per RL step) and `reuse_sampler_allowed`
(bit-identical over 300 steps, update -14.5 %). r88 (U18): no effect, parked.

RULES v3.6 (INBOX 11-16 22:35): a record seed over budget only from idle exit time counts as it is (QUESTIONS when it happens).
**Next decisions:** r102's three seeds against 0.281 (then the five-seed record attempt if the three-seed rule is
met); r101's remaining seeds, paired by seed; the native evaluation-path test; the 11-28 self-running box (queue,
staging, schedules, `docs/research/box_1128.md`) by about 11-22; the 11-25 box's queue, planned on 11-24.

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
| D | `records/v3/track_d/007_dm_math_sil64c05_batched_muon56` | 0.2644 ± 0.0076 | 0.251 | **holder** (Welch vs D 006 p = 0.0226; self-imitation) |
| D | `records/v3/track_d/006_dm_math_fp32_shape05_lr6e-6_sil64c05` | 0.2444 ± 0.0158 | 0.229 | held before D 007 (Welch vs D 005 p = 0.045; self-imitation) |
| D | `records/v3/track_d/005_dm_math_fp32_shape05_lr6e-6` | 0.2150 ± 0.0287 | 0.201 | held before D 006 (Welch vs D 004 p = 0.0034) |
| D | `records/v3/track_d/004_dm_math_fp32_h200_warm` | 0.1552 ± 0.0217 | 0.159 | held before D 005 |
| D | `records/v3/track_d/003_dm_math_fp32_h200` | | | failed: seeds 0 and 2 over budget |

The H100 board closed on 10-24 (A 001 0.2481; D 002 0.1512 ± 0.0202).

Promotion lines (the holder's dev mean + 0.03): **A 0.459 (kept dev, A 007's 0.429), D 0.281 (D 007's 0.251)**.
Three-seed form: every seed at or above the line, or +0.03 over its paired gate-screen seed, none collapsed. Five-seed
form: mean at or above the line, at least 4 of 5 seeds above the holder's dev mean, at most one collapsed. The gating
screen runs on the H200.

## AWS boxes (INBOX 11-10 10:40: back on, screens only, about 50/50 A/D)

| block (EST) | GPUs | ssh | launch template |
|---|---|---|---|
| 11-18 06:30 to 11-19 06:30 | 8x H100 | ended about 06:00 EST 11-19 (terminated) | `rlvr-box-cr-0261a6f8bb30fc123` |
| 11-25 06:30 to 11-26 06:30 | 8x H100 | zjason@174.129.98.131 | `rlvr-box-cr-03ab208e99ca6ca06` (self-running version 1) |
| 11-28 06:30 to 11-29 06:30 | 8x H200 (the maintainer may take it) | zjason@100.29.118.132 | `rlvr-box-cr-08927241900b5884c` (self-running version 3; version 2 is the old one) |
| 12-09 06:30 to 12-10 06:30 | 8x H100 | zjason@174.129.98.131 | `rlvr-box-cr-0813172fa29139b78` (self-running version 1) |

- No scheduled launch exists yet for any block. Until one does, from the block start: `aws ec2 describe-instances
  --region us-east-1`, then `aws ec2 run-instances --region us-east-1 --launch-template LaunchTemplateName=<template>`,
  retried every couple of minutes. The $500 budget stays suspended.
- INBOX 11-18 13:35: by 11-25, make the 11-28 box launch and run by itself, test it for a few dollars and write it up
  in `docs/research/box_1128.md`. State at 17:55 on 11-18 (LOG 11-18 17:55):
  - Built and tested on a cheap instance: `scripts/aws/box_auto.sh` (roles, bucket rules, template, schedules, dry
    run, show), `scripts/aws/box_s3.sh` (stage, check, status, pull, clear) and the on-box scripts `box_boot.sh`,
    `box_runner.sh`, `box_queue.py`. A schedule under role `rlvr-scheduler` launched a t3.large, the box fetched its
    code and queue from S3 and started workers, results and saved states went to S3 by themselves, and a second
    instance carried on after I terminated the first. A launch while an instance holds the box's network interface is
    refused, so a launch by hand cannot clash with a scheduled one.
  - 11-18 18:21: the three future blocks' templates have a self-running default version (`box_auto.sh template`),
    and the dry run of each launch under `rlvr-scheduler` passes (LOG 11-18 18:23). With nothing staged in the
    bucket such a box comes up like the 11-18 one and is run by hand.
  - Still to do: for each block the queues and plan, `box_s3.sh stage`, `home` and `check`, and the schedules
    (`box_auto.sh schedule`); `docs/research/box_1128.md`; the check that my usage and login last through each
    block. Do not run `scripts/aws/box_launch.sh` for a reservation set up with `box_auto.sh`.
  - Left on AWS: template `rlvr-box-test`, roles `rlvr-scheduler` and `rlvr-box`, the test's output under
    `box/rlvr-box-test/out/` in the bucket (run `box_s3.sh clear rlvr-box-test` before a re-test). No schedules.
- **AWS charges made by the agent** (outside the blocks the maintainer bought), corrected 11-19 06:32 from the
  bucket's final contents (LOG 11-19 06:32): the 11-18 self-launch test about $0.03; S3 for the 11-18 box about
  $0.27 (saved states 25.07 GB for 7 days $0.135, results 0.355 GB for 90 days $0.025, 92 state uploads about
  $0.07, result pushes about $0.04, my estimate). Running total: about $0.30, at most about $0.40. The earlier
  "about $2.29 at most" allowed $1.72 for track A states that were never sent. I have not read the bill.
- INBOX 11-18 13:39: the 11-25 and 12-09 blocks are new. Both show `scheduled` (checked 11-18 14:17); each needs a launch with
  its own reservation id, a 24 h queue and a packing checked against 80 GB. Use the self-starting setup on 11-25 if
  it is ready.
- While a box is up: wake every 20-30 min, `box_sync.sh from` every wake and a last time before
  06:00 EST on the end day, and bring resume state back (INBOX 11-04).
- The plan for each box is in PLAN ("Box days") and LOG 11-10 10:33.

## Next

- The 11-18 box ended at about 06:00 EST 11-19 (LOG 11-19 06:32). Every finished result is in the checkout: 33
  runs ok and 13 FAILED lines in its `queue/done.txt`. What it leaves for the next boxes:
  - One native D run per H100 at 65,536-token micro-batches is the planning number: 0.61 to 0.64 s a step, 11,441
    steps in 2.1 h and 14,180 in 2.5 h. A pair of 32,768 runs on one GPU, the second started after the first's
    first evaluation, gave 11 to 22 % more steps per GPU with each run 1.7-2x as long (a pair started together
    runs out of memory). Which to use is decided with the 11-25 queue on 11-24.
  - The native backend resumes from its saved state, and a push loop on the box gets results and states to S3 by
    itself.
  - r85: `a007_ent15` +0.0222 over three pairs (the track A candidate; needs five seeds or more). r86: the Muon
    arms are level with or below the AdamW control at step caps.
  - Six unfinished runs' states are in `box-state/box1118/` in the bucket and expire from 11-26. On 11-24 decide
    whether any continues on the 11-25 box (copy its state to that box's prefix first). Results and logs stay
    under `box/box1118/` for 90 days (`box_s3.sh pull box1118 <tag> <dir>`).
- r104, the native evaluation-path test: **done (LOG 11-19 09:04); the evaluation is cleared.** D 007's five
  final checkpoints scored on dev by both samplers on the same weights (B200): greedy, native minus vLLM is
  -0.0013 (SE 0.0013; 2,860 of 3,000 completions identical, 10 solved by vLLM only, 6 by native only); at
  T = 1 the solve rate differs by +0.0003 (SE 0.0012) and the entropy per token by a factor of 1.0006. So the
  box's equal-step gap (about -0.01 for native, three pairs of finals) is training under the native update
  settings or chance, not evaluation, and the samplers draw alike at these weights. No GPU time goes to the gap
  by itself. Next: read in the code whether the logged training entropy counts the same rows and mask under
  `share_group_prompt` / `reuse_sampler_allowed` / 65,536-token micro-batches; if it does, a few short
  step-capped runs switching one setting at a time. Result: the item's `seeds/0/sampler_paths.json`.
- A runs fill the H100 (79.1 of 79.18 GB) and three ran out of memory at `vllm_gpu_memory` 0.35 (control seeds 0 and
  1, `cos467` seed 0). Since 17:47 every A item not yet started uses the `_h100m30` configs (share 0.30, about 4 GB
  more for the trainer; same maths, read together with the arm's `_h100` seeds), and the three failed runs resume
  under them from their saved states. The `_h100m30` runs stayed inside the card (ten finished), so the 11-25
  and 12-09 A arms use share 0.30.
- D on the H100: two runs per GPU take 5.2-5.6 h for 11,441 steps and 6.2-6.8 h for 14,180; one run alone takes about
  3.65 h and 4.5 h (LOG 11-18 17:55).
- r101 (B200, RL only from D 007's checkpoints, queue `screens_rl_b200`, seeds 0-2, not gated):
  `d007rl_native_b200` (native sampler, `reuse_sampler_allowed`, shared prompt, 65,536-token micro-batches) against
  `d007rl_ctl_b200`, both 17,400 s. Done: native seed 0, 0.3033 after 34,893 steps; control seed 0, 0.2483 after
  18,215; control seed 1, 0.2383 after 20,032. Running since 08:49: native seed 1 (29626073) and control seed 2
  (29618971), both on node5602-c1. Queued: native seed 2, for 29632479 (preemptable, room for two). Read
  paired by seed (steps and final dev pass rate), with the node each ran on.
  **Node speed (LOG 11-19 09:04).** Native seed 0 was still climbing at the end (+0.0023 dev per 1,000 steps from
  step 15,000) and ran a third faster from about 01:20 EST, within three minutes, when other jobs on its node
  ended (31.3k to 41.5k tok/s; update -31 %, sampling -16 %; the same tokens per step). Cause unknown. A
  time-budgeted native run's steps can move by a quarter to a third with its node (0.02-0.03 dev), so native
  results carry steps and node, and arms are compared at step caps. `scripts/research/node_trace.sh` (aa41eb8)
  writes `seeds/<seed>/node_trace.csv` (the job's GPU and node-wide readings every 30 s) for items with a
  `node_trace` file: r101's remaining items and r102's seeds; never records. It works on a GPU node (first rows
  08:49 on node5602-c1; pressure columns empty there). When r101's two running items end (about 13:45), read each
  trace against its train log by time (GPU power and clock, the job's CPU clock, the spin time, load).
  r97 and r99 (B200, end to end) are superseded by r102: their preemptable workers 29497701/02/05 stay held (the
  cap of four running preemptable GPUs goes to r102 and one r101 worker).
- Track A: A 007's seeds reached step 467 with 1,700-2,700 s of the budget unused (room for about 50 more steps),
  and a step is about 55 % sampling, 45 % update. r103 (three timing runs on the H100, LOG 11-19 06:32): neither
  same-maths option helps on an 80 GB card as it is. `constrain_batched` samples 1.16 to 1.27 times slower per
  token (its mask is built on the CPU, 26 to 35 % of its sampling time; a mask builder that cost nothing would
  save at most 20 % of sampling). No checkpointing with the shared prompt needs 4,096-token micro-batches to fit,
  and its update is then 1.73 to 1.89 times slower per token. The backward pass is about two thirds of the update;
  sampling is vLLM's `generate` (99.5 %). Leads: a faster mask builder; no checkpointing at 8,192 on an H200
  (141 GB). Next arm: more steps with `a007_ent15`'s setting at five seeds or more, decided with the 11-25 queue
  on 11-24.
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
