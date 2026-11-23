# Status

(maintained by the research agent; see AGENT.md)

Updated 2026-11-23 02:53 EST. The run ends 2026-12-19. The agent runs on Opus 5.5.

**Low-usage window until 2026-11-24 15:00 EST, then again from the 11-25 box start until 11-25 18:00 (INBOX 11-22).** In the window each wake does only the essentials (read finished results into the LOG, requeue failed items, keep the GPUs fed) and ends the loop without a wakeup; the loop is re-issued after about 5 h idle. From 11-24 15:00 to the box start the wakes run at the normal pace for a full replan of the 11-25 box, restaged in S3 and checked. The sampler-mask speed item (the next item after r128's read) waits for the window to end: no code for it before then. **D 008's record attempt is running (LOG 11-22 16:28, 21:41 and 11-23 02:53).** Seed 0 ended at dev 0.2950, test 0.3188. Seed 4 runs on 30099011 (since 22:28) and seed 2 on 30132623 (restarted at 02:07 after a preemption; the partial run is in extra/). Seeds 1 and 3 were preempted at midnight and restart when 30099012 and 30132625 (pending) start. r116 ended seeds 0-2 at +0.0367, +0.0450 and +0.0367 over r101's control; its own control (r84's weights) ended seeds 0-1 0.0550 and 0.0767 under r116, and control seed 2 waits for 30165584 (normal B200, pending). FineMath-4plus shards 3-31 (8.3 GB) are in scratch. Five GPUs. Next wake: read control seed 2 and D 008's seeds as they end; requeue lost items once; from 11-24 15:00, replan the 11-25 box.

**The speed stack turns into score (LOG 11-19 02:05, 11-19 14:35 and 11-20 09:23).** r101 (B200, RL only from D
007's checkpoints, 17,400 s, not yet gated): the native arm's three seeds ended at dev 0.3033 (34,893 steps),
0.2883 (44,862) and 0.2833 (44,987), mean 0.292; the control's three at 0.2483, 0.2383 and 0.2750 (17,608 to
20,032 steps), mean 0.254. Pairs: +0.055, +0.050 and +0.008 (mean +0.038, SE 0.015). At equal steps the arms are
level (+0.012, +0.005 and -0.016), so the gain is the extra steps: around step 29,000 the native seeds stood at
0.282, 0.269 and 0.263, and seeds 1 and 2 gained about half a point from 35,000 to 43,000.
**r102 is its H200 promotion screen**: `calibration/research_r102/d007_speed_h200` (D 007 end to end with the
fused pretraining for 46 min and the native RL settings for 18,000 s), seeds 0-2 in `queue/screens_r92.txt` on
H200 preemptable workers 29450693/94/702 (released 01:52 on 11-19). Seed 0 started at 09:40 on 11-20 (node4802);
its fused pretraining runs at about 473k tokens a second, so 46 min gives about D 007's 56-minute checkpoint. Seed
1 started at 09:54 (node4603) and was preempted at 10:04 in its pretraining; the requeued worker redoes it, but
the requeue reset its age (priority 31,298). Seed 0 ended at 0.270 after 38,914 steps (2.16 steps a second; best 0.2817), +0.019 over D 007's dev mean and under the line (LOG 11-20 15:32). Seed 2 ended at 0.300 after 42,588 steps (2.36 steps a second on node4601; best 0.3217), above the line (LOG 11-21 20:52). Seed 2's first worker failed it in the pool quota failure of 11-20 15:15; it is requeued with worker 29808853. Read against D's line
0.281 (D 007's dev mean 0.251 + 0.03) together with the steps each seed reaches: at the H100's 0.62 s/step it
would reach about 29,000, where the B200 seeds are near the line and not clear of it (expected finals about 0.27
to 0.29). If the three-seed rule is met, start a five-seed record attempt (r102's config without `no_save` on
preemptable H200 workers, or r114's 6 h form below). Seed 0's pass from pretraining into native RL (never run end
to end before) is due about 10:30. On the 11-18 box the native sampler came out a little behind vLLM on dev at
equal steps, by about 0.01 (finals -0.021 over three finished pairs, SE 0.009; evaluation curves -0.008 over five
pairs, SE 0.003; LOG 11-19 06:32), against about twice the steps in the same time. r104 then scored D 007's final
and pretrained checkpoints with both samplers on the same weights, and they agree at both ends (LOG 11-19 09:04
and 11-20 04:12), so the gap is put down to drift between runs and is closed until r102's end-to-end numbers.
INBOX 11-19 01:32 (critique of the box plan) is answered in LOG 11-19 02:05, and its seeds point was replaced by
INBOX 11-19 14:29 (LOG 11-19 16:51): speed stack first, on Engaging; on the boxes bold arms at two or three seeds
against a five-seed control, each arm's expected effect and the control's seed sd written first, the top arms
re-run before anything is built on them. A standby queue is staged for all three boxes; each is planned again the
day before (11-24, 11-27, 12-08). INBOX 11-19 16:28: AWS and Engaging are assumed to run at different speeds until
the same item has been timed on both, so a box's timings and time-budgeted results are read only against that
box's own control.

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

**Jobs (11:06 EST 11-22).** Now: 30069082 (normal B200, node5700-c1) runs r116 seed 1 since 09:51; preemptable H200 30051396 runs r120 seed 2 on node4803 since 06:27; preemptable H200s 30073610 and 30073611 are pending for r102 seeds 3 and 4 (LOG 11-22 11:06); 4 GPUs. r128 ended (LOG 11-22 11:02). Earlier: Workers write to scratch (`CKPT_DIR` in `--export`) since the pool's file server
filled up at 18:27 on 11-20 (LOG 11-20 18:42; QUESTIONS 11-20 18:42). The pool shows room again (64.5 %), but the
workers stay on scratch until a write probe there passes. Scratch holds 3 % of its space and 74.7 % of its 1.0M file
limit (12:30 on 11-21, up from 70 % at 07:29; not from these runs' outputs, a few files a seed; above 85 % the
workers go back to the pool after a write probe). 29966174 (normal B200, node5501-c1) ran r117 p128
seed 0 from 17:02 to 21:53 (LOG 11-21 21:58). 30010825 (normal B200, node5501-c1) ran U25's 3x rung from 21:59 to 03:32: no, so 6x goes to the 11-25 box (LOG 11-22 03:52). 30028921 (normal B200, node5700-c1) ran r122 and r123 from 23:19 to 23:43: r119 is closed (0.008 s from update 200 on), compile_update failed its check and gained under 0.03 s, so it stays off, and r124 (the update's synchronisations) is next (LOG 11-22 00:00). r124 (`update_host_index`, off by default; LOG 11-22 00:25) ran on normal B200 30051395 (node5501-c1, 03:50 to 04:08): the same maths, but its paired gain missed the rule, so it stays off (LOG 11-22 04:13; preemptable 30039308 was cancelled for the freed normal slot, LOG 11-22 03:52). Its first submission, 30024430, stopped at once on the profiled config's max_steps 300 (LOG 11-21 22:54). Normal B200 30052724 (node5700-c1, 04:31 to 04:44) ran U25's one-GPU data-parallel check: it meets its rule (LOG 11-22 04:48). Normal B200 30054646 (node5700-c1) ran r116 seed 0 from 04:50 to 09:41: +0.037 over its control at the final (0.340 vs 0.303), ahead at all 98 matched-time and 69 matched-step evals, completion length about 24 to 25 tokens; the rule waits for seed 1 (LOG 11-22 09:49). 30033104 (normal B200, node5601-c1) ran r105 lr 24e-6 seed 2 from 03:35 to 08:27 (LOG 11-22 08:31). Normal B200 30065238 (node5601-c1) ran r125's capture check and two profiles from 08:30 to 08:47: met its rule (LOG 11-22 09:06). Preemptable H200 30067245 (node4601) ran it again for r126 from 09:11 to 09:17: the maths held, the timing did not (LOG 11-22 09:37). 30000622 (node5701-c1) ran r116's two pretraining items from 18:08 to 19:25, and 30008197 (node5601-c1) ran U25's D 007 base probe and 3x smoke run from 19:33 to 19:42 (LOG 11-21 19:57). 29999304 (normal B200) ran r119's profiles and r116's data from 17:50 and exited (LOG 11-21 18:16). 29966173 ran r105 lr 24e-6 seed 1 and exited at 17:29; 29942141 ran r105 lr 12e-6 seed 2 and exited at 16:55. Preemptable B200 29999306 (`screens_rl_b200`) started r116 seed 1 six times between 05:56 and 09:43 and was preempted each time; it was cancelled while pending and normal B200 worker 30069082 was submitted for seed 1 at 09:49, ahead of U25's two-GPU check (LOG 11-22 09:49). Preemptable H200 30068951 runs r127's finer-bucket capture check (LOG 11-22 09:44) (r117 p128 seed 1 was taken out at 21:58).
29837209 and 29837211 ran r115 p32 seeds 0 and 1 (LOG 11-21 12:40) and exited; 29963843 ran the sampler timing
(12:28 to 12:37). r117 p128 seed 1's preemptable worker 29837212 was cancelled at 17:13 so that the preemptable slots go to H200 work (LOG 11-21 17:13). r118 (preemptable B200 29980436, 80 s) found the update step overhead-bound: GPU busy 0.40 of its wall time,
11,689 kernel launches per update (LOG 11-21 14:59). r119 (29999304), the same profile with PyTorch's
memory-efficient attention kernel against the default, did not meet its rule: 0.16 s off the unprofiled update but
0.009 s off the profiled one, so capture or compile is next, built on the efficient kernel (LOG 11-21 18:16). Both profiles
ran 20 updates at the start, while a real run's late update takes 0.17 to 0.20 s, so r122 timed 600 updates: the
0.16 s does not last past start-up (INBOX 11-21 22:10, LOG 11-21 21:58 and 11-22 00:00).
The three CPU jobs for the data (29965651 on node3107, 29980655 on node3409, 29981583 on node2702) hung or failed
their start checks; a diagnostic (29982547) found that CPU nodes do not reach the pool, while home and scratch
respond. On `screens_r92`: preemptable H200 29837214 ran r102 seed 2 on node4601 from 14:40 to 20:28 (dev 0.300, LOG
11-21 20:52), and 30006765 ran r102 seed 1 on node4602 from 19:25 to 01:12 (dev 0.310; three-seed mean 0.293, LOG 11-22 01:14; 29837213 restarted on node4301 at 18:20, found the venv
unreadable and exited; LOG 11-21 19:57). On `gate_r120_h200`: preemptable H200 29995938 ran r120 seed 0 on node5000 from 18:38 until it was preempted at 23:19, about 3.9 h into the RL part (no resume state: RL screens are no_save); Slurm requeued it, and when it starts again the worker moves the partial run to `extra/` and reruns seed 0 from the start, so r120's seed 0 read moves about 6 h past that start, and 30017487 (pending since 20:52) is for seed 1; preemptable H200 30051396 (submitted at 03:47) is for the next item, seed 2, then r102's seeds 3 and 4 (queued at 01:14). r124 (update_host_index's check with paired timing, then two profiles; rule LOG 11-22 00:25) is normal B200 30051395, running since 03:50. Held: 29497701/02/05 (B200
preemptable; r97/r99, superseded by r102) and 29445218 (H200; r90). My jobs wait on age (298 an hour; LOG 11-19
21:02 and 11-19 23:29); a requeue resets it. **Practice (11-20 04:34):** one pending worker for each queued long
item, submitted when the item is queued, and never chained with `--dependency` (a job waiting on a dependency or a
hold gains no age here). Below 50,000 the scheduler plans no start (`bf_min_prio_reserve`), so preemptable workers
start only when a GPU is free as it tries them. The B200 workers exclude only the drained `node5500-c1`; the H200
workers exclude node5100 (the pool filesystem hangs there), node2500 and node4900 (slow) among others.
**Maintenance: every node is reserved from 00:00 to 21:00 on 12-15**; a 6 h worker must start by 18:00 on 12-14. At
most 2 normal and 4 preemptable GPUs run at once, the cap of six.

**Settled speed settings for new D arms (LOG 11-17 and 11-18; timed on B200; each is the same maths and off by
default).** Pretraining: `--fused-loss --micro-batch 128` (+21 % tokens per second; 46 min gives about the old
56-min checkpoint). RL: the shared prompt with 65,536-token micro-batches (update 14-18 % shorter; fits the H200),
the native sampler (`rollout_backend: "native"`, 1.40-1.60x faster per RL step) and `reuse_sampler_allowed`
(bit-identical over 300 steps, update -14.5 %). r88 (U18): no effect, parked. Fused pretraining at micro-batch 128
holds 52,866 MiB on a B200 (r107, LOG 11-20 04:12), so it fits an 80 GB H100. `factored_grammar` is not one of
these: timed in r106, same maths, no speed at trained weights.

RULES v3.6 (INBOX 11-16 22:35): a record seed over budget only from idle exit time counts as it is (QUESTIONS when it happens).
**Next decisions:** workers go back to the pool only after a write probe there passes (QUESTIONS 11-20 18:42);
r102's seeds 3 and 4 (queued on `gate_r120_h200` after r120's seeds 1 and 2; seeds 0-2 ended at 0.270, 0.310 and
0.300, mean 0.293, LOG 11-22 01:14) for the five-seed form: mean at least 0.281, four of five above 0.251, at most one
collapsed (LOG 11-21 20:52); U25's data-parallel check on two B200s (the one-GPU form met its rule, LOG 11-22 04:48) the first time both normal slots are free, by 11-24; r120 (r102's H200 gate with lr 1.2e-5, after r105
met its rule) seeds 0-2 against the rule written first in LOG 11-21 17:13 (all three at 0.281 or more: a
five-seed record attempt; a mean of 0.276 or more: seeds 3 and 4), paired with r102's seeds; r105's lr
24e-6 is closed for the native regime (seed 2 +0.010 at matched steps, three pairs +0.0144, short of 0.015; LOG 11-22 08:31), so the gates keep lr 1.2e-5; U25's
3x rung said no (LOG 11-22 03:52), so its 6x rung goes to the 11-25 box; r113's three arms are all between and left to the box (sil256
the strongest, stack3 the weakest); r116's
two RL items (queued on `screens_rl_b200`; pretraining done, LOG 11-21 19:57) against its rule (LOG 11-21 12:40); the speed
work, first among the code items since INBOX 11-21 22:10 (LOG 11-21 21:58 and 11-22 00:00: r122 and r123 are done, closing r119 and compiling the layers; r124
took most synchronisations out of the update (63 to 18 an update) with the same maths, but its paired gain of 0.006 s missed its rule, so it stays off (LOG 11-22 04:13); capture (r125, LOG 11-22 06:32) met both parts of its rule (median gain 0.018 s, 35/49 faster, p 0.002; the same maths within its bounds), so new configs turn it on with update_host_index, after the same check on an H200 (LOG 11-22 09:06); on an H200 (r126, LOG 11-22 09:37) the maths held but capture was slower (median gain -0.022 s, 17/49 faster), so gate and record configs keep it off and B200 screens may turn it on; its padding count (0.196 of the bucket tokens) made finer buckets the next item, and they met capture's rule on the H200 (r127: faster in 49 of 49 warm pairs, median 0.012 s; LOG 11-22 10:21), so new H200 gate and record configs turn capture on with the fine steps (16, 8, 8); r128 (LOG 11-22 11:02) profiled a step with fine capture on an H200 (418 ms: sampling 173 ms, update 189 ms): the sampler mask's CPU time is the largest item (104 ms a step, against 7.9 ms for the head's forward, 13.3 ms for replay_batch and at most 11.4 ms for attention), so the native sampler's mask is the next speed item (same masks, so no loss comparison; its rule comes first); sil's 64 replayed rows (54 ms) come later; U25's data-parallel pretraining and resume
(needed now that the 3x rung said no; code in, off by default, LOG 11-22 04:03; its GPU check met its rule on one B200 (LOG 11-22 04:48) and runs on two when r116 seed 1 has ended, about 15:00 on 11-22, as the 11-25 box is the date); U26 after the speed items) (LOG 11-21 12:40): the sampler timing put the answer grammar and stop checks at 9 to 18 % of a decode step at
trained weights, so the update comes first, with a timing item (the policy rows' forward and backward, the
self-imitation rows, the optimizer step) before any code (r118 found it overhead-bound: GPU busy 0.40, 11,689 kernel launches per update; r119's memory-efficient
attention kernel did not meet its rule, so capture or compile is next, on that kernel; LOG 11-21 14:59 and 18:16), each change the same maths and off by default (r112 closed model size; a 90 s
step rate follows the starting weights, so the timing item compares runs from one checkpoint on one node and logs
each batch's longest completion, LOG 11-20 11:48); each box's standby queue (staged and checked for all three
blocks, `docs/research/box_1128.md`) planned again the day before its block (11-24, 11-27, 12-08), with sil256 the
strongest r113 arm so far and track A arms included if a bold one is ready; each worker's time from submission to
start; and, in the idle wakes before 11-24, bold end-to-end track D arms (U17, U19) for the 11-25 box.

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

- **Scheduled launches exist for all three blocks** (created 11-19 about 10:25; LOG 11-19 10:47): 11 one-time
  EventBridge schedules per block, `rlvr-box-<reservation id>-m0001` to `-m1080`, at 1, 5, 10, 25 and 60 min and
  3, 6, 9, 12, 15 and 18 h after the block starts. Each calls RunInstances with the block's template (its default,
  self-running version) under role `rlvr-scheduler`. Check: `bash scripts/aws/box_auto.sh show <reservation id>`
  (11-19 10:32: reservations `scheduled`, 11 schedules `ENABLED` each, no instance). A launch while an instance
  is up is refused, so nothing is launched by hand; if a box dies, a later schedule starts a new one, which
  carries on from the bucket. **Before terminating a box on purpose inside its block, run `box_auto.sh
  unschedule <reservation id>`.** **A standby queue is staged for every block (11-19, below) and each stage reads
  "check ok"; a re-stage must be in place before its launch:** a box that finds no `code.tgz` in its first 3 min
  comes up like the 11-18 one and is run by hand. The $500 budget stays suspended.
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
  - 11-19 10:47: the schedules exist (above). `box_s3.sh` now also stages the token file of a screen that
    pretrains (one tar under `box/home/`, unpacked in place on the box), so end-to-end D screens can be queued on
    a box; `tests/test_box_s3.py`. The templates' disks are 2,000 GB. A failed end-to-end item starts again from
    pretraining (no saved state).
  - 11-19 11:08: **the cheap fetch test passed** (LOG 11-19 11:08). A t3.large launched by a schedule fetched a
    directory, a one-file tar and the tokenizer from `box/home/` (25 MB); two stub items checked five files
    by sha256, ran, failed once as planned, resumed and pushed their results; the box shut itself down
    3 min 48 s after the launch. These were also the first runs of `stage`, `home` and `check` with the 10:47
    code. The tokenizer is now in `box/home/`. **Not yet tested:** the 4.64 GB token-file tar (not uploaded), a
    real checkpoint or the model cache on the new boxes, anything with a GPU.
  - 11-19 12:28: **a failed fetch is now reported** (344eb32; LOG 11-19 12:28). In `box_boot.sh` a copy, sync or
    untar that fails marks its path as not fetched even when part of it arrived. The cheap test was repeated
    with a cut-off tar among four paths: the box's status file named it (`NOT FETCHED: ...`), the other three
    arrived, both stub items passed and the box shut itself down 3 min 46 s after the launch. A file that is
    complete but wrong is not caught on the box; `box_s3.sh check` looks at the stage before a launch.
  - 11-19 12:40: **`docs/research/box_1128.md` is written** (LOG 11-19 12:40): what exists on AWS, what a box does
    by itself, a read-only check for each piece, the rules for working by hand, the tests and what is not
    tested. It gets the staged commit, queues, plan and `check` output per block when each stage is made.
    **Limit:** a track D screen with its own pretraining keeps no saved state on a box (it runs through
    `run_seed_v3.sh`); after a lost instance it starts again. Track A and RL-only D screens resume.
  - 11-19 16:51: **standby queues are staged for all three blocks** (INBOX 11-19 14:40; LOG 11-19 16:51), at
    commit 849c85c, 20:22-20:23Z on 11-19; `box_s3.sh check` reads "check ok" for each (16:16 EST). 11-25: box
    `rlvr-box-cr-03ab208e99ca6ca06`, `queue/box_1125.txt`, batch r108. 11-28: `rlvr-box-cr-08927241900b5884c`,
    `queue/box_1128.txt`, r109. 12-09: `rlvr-box-cr-0813172fa29139b78`, `queue/box_1209.txt`, r110. Each: 47
    lines, one native RL-only track D run per GPU from D 007's pretrained checkpoints (in `box/home/` since 15:47
    EST), a five-seed control and nine bold arms (twice the RL time, self-imitation x4 two ways, mini levels, hard
    levels doubled, lr 2x and 4x with cosine decay, 16 prompts x groups of 32, a three-change stack). In 24 h:
    five control seeds, two long2x seeds, three seeds of seven arms, two of lr24cos; a tail runs until the block
    ends. All track D: no bold track A arm is ready, so the 50/50 split of INBOX 11-10 is taken up again on each
    planning day. The five r111 smoke runs passed (LOG 11-20 10:26). Each stage is replaced
    on its planning day (11-24, 11-27, 12-08).
  - Still to do: each block's queue planned again the day before (11-24,
    11-27, 12-08) and re-staged with `box_s3.sh stage`, `home` and `check` (new arms in new config batches; track
    A arms and end-to-end track D arms join if ready); the check that my usage and login last through each block;
    for 11-28, whether a seed's pretrained checkpoint can be sent to S3 so a replacement instance repeats only the
    RL part, and whether two runs fit one H200. Before any re-test run `box_auto.sh template test` (the test
    template's block end is 3 h after it is made) and `box_s3.sh clear rlvr-box-test`. Do not run
    `scripts/aws/box_launch.sh` for a reservation set up with `box_auto.sh`.
  - Left on AWS: template `rlvr-box-test` (default version 3), roles `rlvr-scheduler` and `rlvr-box`, and the 33
    schedules above (free). In the bucket: the repeat test's stage and output under `box/rlvr-box-test/`, the
    first 11-19 test's under `box/rlvr-box-test-1119/`, the 11-18 test's under `box/rlvr-box-test-1118/`, and the
    test files under `box/home/orcd/pool/boxtest/` (a cut-off `bad.bin.tar` among them; the others also under
    `~/orcd/pool/boxtest/` on Engaging, 21 MB). The test stages and test files go once the build is audited. Also
    in the bucket since 11-19: the three blocks' stages under `box/rlvr-box-<reservation id>/in/` and, under
    `box/home/`, the tokenizer and D 007's five pretrained checkpoints (1.04 GiB).
  - The 10:47 question (may an end-to-end screen on the 11-28 H200 box serve as the gating screen?) is answered
    (INBOX 11-19 14:32): the decision is mine, to be made when the time comes with the reasoning in the LOG. The
    maintainer leans to speed work on Engaging, and says to assume AWS and Engaging run at different speeds unless
    shown otherwise (INBOX 11-19 16:28). For now r102 stays the gate on Engaging and the standby holds no
    end-to-end screen.
- **AWS charges made by the agent** (outside the blocks the maintainer bought), corrected 11-19 06:32 from the
  bucket's final contents (LOG 11-19 06:32): the 11-18 self-launch test about $0.03; S3 for the 11-18 box about
  $0.27 (saved states 25.07 GB for 7 days $0.135, results 0.355 GB for 90 days $0.025, 92 state uploads about
  $0.07, result pushes about $0.04, my estimate); the two 11-19 fetch tests about $0.01 each (LOG 11-19 11:08 and
  12:28); D 007's checkpoints kept in `box/home/` for the boxes about $0.07 over their 90 days (1.04 GiB; LOG
  11-19 16:51). Running total: about $0.39, at most about $0.50. The earlier "about $2.29 at most" allowed $1.72
  for track A states that were never sent. I have not read the bill.
- INBOX 11-18 13:39: the 11-25 and 12-09 blocks are new. Both show `scheduled` (checked 11-18 14:17); each needs a launch with
  its own reservation id, a 24 h queue and a packing checked against 80 GB. Use the self-starting setup on 11-25 if
  it is ready.
- While a box is up: wake every 20-30 min, `box_sync.sh from` every wake and a last time before
  06:00 EST on the end day, and bring resume state back (INBOX 11-04).
- The plan for each box is in PLAN ("Box days", revised 11-19), LOG 11-19 16:51 (the standby) and
  `docs/research/box_1128.md`.

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
- r104, the native evaluation-path test: **done (LOG 11-19 09:04); the evaluation is cleared.** D 007's five final
  checkpoints scored on dev by both samplers on the same weights (B200): greedy, native minus vLLM is -0.0013 (SE
  0.0013; 2,860 of 3,000 completions identical, 10 solved by vLLM only, 6 by native only); at T = 1 the solve rate
  differs by +0.0003 (SE 0.0012) and the entropy per token by a factor of 1.0006. So the box's equal-step gap
  (about -0.01 for native, three pairs of finals) is training under the native update settings or chance, not
  evaluation, and the samplers draw alike at these weights. No GPU time goes to the gap by itself. The code read
  is done (LOG 11-19 09:15): the logged entropy is the same quantity on the same rows and mask in both arms, there
  is one optimizer step per GRPO step whatever the micro-batch size, and the reused allowed sets and shared prompt
  are tested equal to the recomputed path. So the update's maths does not explain the gap, and no
  one-setting-at-a-time runs are needed. Left: drift between runs, or a sampler difference at early weights.
  **Second item done (LOG 11-20 04:12)**: D 007's five pretrained checkpoints, both samplers. At T = 1 the entropy
  per token is native / vLLM = 0.9981 (log-prob +0.0080, SE 0.0032; solve rate -0.0003, SE 0.0002). Greedy pass
  rates are near zero in both (0.0010 and 0.0013), with 2,232 of 3,000 completions identical (near-ties at
  untrained weights, my reading). The training gap was native 5 to 9 % higher in entropy; this is 0.2 % lower. By
  the rule the sampler is cleared at both ends, the gap is put down to drift, and the question is closed until
  r102's numbers. Results: each item's `seeds/0/sampler_paths.json`.
- A runs fill the H100 (79.1 of 79.18 GB) and three ran out of memory at `vllm_gpu_memory` 0.35 (control seeds 0 and
  1, `cos467` seed 0). Since 17:47 every A item not yet started uses the `_h100m30` configs (share 0.30, about 4 GB
  more for the trainer; same maths, read together with the arm's `_h100` seeds), and the three failed runs resume
  under them from their saved states. The `_h100m30` runs stayed inside the card (ten finished), so the 11-25
  and 12-09 A arms use share 0.30.
- D on the H100: two runs per GPU take 5.2-5.6 h for 11,441 steps and 6.2-6.8 h for 14,180; one run alone takes about
  3.65 h and 4.5 h (LOG 11-18 17:55).
- r101 (B200, RL only from D 007's checkpoints, queue `screens_rl_b200`, seeds 0-2, not gated; LOG 11-19 02:05,
  11-19 14:35 and 11-20 09:23): `d007rl_native_b200` (native sampler, `reuse_sampler_allowed`, shared prompt,
  65,536-token micro-batches) against `d007rl_ctl_b200`, both 17,400 s. Done, all six: native seed 0, 0.3033 after
  34,893 steps (node5600-c1); native seed 1, 0.2883 after 44,862 (node5602-c1); native seed 2, 0.2833 after 44,987
  (node5600-c1); control seed 0, 0.2483 after 18,215; control seed 1, 0.2383 after 20,032; control seed 2, 0.2750
  after 17,608. Pairs +0.055, +0.050 and +0.008 (mean +0.038); at equal steps +0.012, +0.005 and -0.016. Native
  finals: mean 0.292, sd 0.010; control finals: mean 0.254, sd 0.019. Native dev around step 29,000: 0.282, 0.269
  and 0.263; from 35,000 to 43,000 seeds 1 and 2 gained +0.008 and +0.006. Native seeds 3 and 4 are not queued
  (the arms pair against seeds 0-2).
  **Node speed (LOG 11-19 09:04).** Native seed 0 was still climbing at the end (+0.0023 dev per 1,000 steps from
  step 15,000) and ran a third faster from about 01:20 EST, within three minutes, when other jobs on its node
  ended (31.3k to 41.5k tok/s; update -31 %, sampling -16 %; the same tokens per step). Cause unknown. A
  time-budgeted native run's steps can move by a quarter to a third with its node (0.02-0.03 dev), so native
  results carry steps and node, and arms are compared at step caps. `scripts/research/node_trace.sh` (aa41eb8)
  writes `seeds/<seed>/node_trace.csv` (the job's GPU and node-wide readings every 30 s) for items with a
  `node_trace` file: r101's remaining items and r102's seeds; never records. **First traces read (LOG 11-19
  14:35):** from 11:00 EST the load on node5602-c1 rose from about 15 to about 153; the vLLM control ran
  35 % slower (0.85 to 1.15 s/step, nearly all in sampling) and the native run 9 % slower (0.35 to 0.39).
  The GPU's clock did not move and no throttle flag was set, so the cause is on the CPU side, and vLLM runs are
  the more exposed. Native seed 0's larger jump at 01:20 (mostly in its update) is not explained by this.
  r97 and r99 (B200, end to end) are superseded by r102: their preemptable workers 29497701/02/05 stay held (the
  cap of four running preemptable GPUs goes to r102 and one r101 worker).
- r105 (B200, RL only, not gated; LOG 11-19 09:28): `d007rl_native_lr12e-6_b200` and `d007rl_native_lr24e-6_b200`
  (r101's native item with lr 2x and 4x D 007's 6e-6; nothing else changed), seeds 0-2. **First pair (LOG 11-20
  09:23):** lr 12e-6 seed 0 ended at 0.2900 after 36,808 steps (node5602-c1) against the control seed 0's 0.3033
  after 34,893 (node5600-c1): -0.005 at matched steps (mean of 7 block differences) and -0.013 at the final, so
  level and not the gain expected. **Second pair (LOG 11-20 14:24):** lr 24e-6 seed 0 ended at 0.2967 after 45,636
  steps (node5800-c1): +0.002 over the 69 shared evals, -0.007 at the final, so also level, with no order by dose.
  The two seed-2 items were dropped for r115 (LOG 11-21 02:00). **lr 12e-6 seed 1 (LOG 11-21 07:39):** 0.3233
  after 48,070 steps (node5702-c1) against the control seed 1's 0.2883 after 44,862 (node5602-c1): +0.0243 over 89
  shared evals (ahead at 85), +0.035 at the final. Two pairs: +0.0096 at matched steps, +0.011 at the final, so
  neither met nor closed. **lr 12e-6 seed 2 (LOG 11-21 17:13):** 0.3067 after 40,426 steps (node5701-c1)
  against the control seed 2's 0.2833 after 44,987: +0.0290 over 80 shared evals, +0.023 at the final.
  **Three pairs: +0.0161 at matched steps (SE 0.0107) and +0.0150 at the finals (SE 0.0146), so the rule is
  met at the bar (seed 0 behind) and lr 12e-6 goes into the next H200 gate config, r120.** **lr 24e-6 seed 1 (LOG 11-21 17:59):** 0.3367 after 47,812 steps (node5501-c1) against the control seed 1's 0.2883: +0.0311 over 89 shared evals (ahead at 88), +0.048 at the final. Two pairs: +0.0166 at matched steps. **Seed 2 (LOG 11-22 08:31):** 0.2950 after 50,132 steps (normal B200 30033104, node5601-c1) against the control seed 2's 0.2833: +0.0100 over 89 shared evals (ahead at 66), +0.012 at the final. **Three pairs: +0.0144 at matched steps (SE 0.0087) and +0.0178 at the finals, short of 0.015, so lr 24e-6 is closed for the native regime; the gates keep 1.2e-5.** Expected: about +0.02 at 2x (range -0.02 to +0.05) against a paired SE of about 0.005. Read
  paired by seed against `d007rl_native_b200` at matched steps (block means of ten dev evals), with steps and
  node. A gain of 0.015 or more over three pairs puts that lr into the next H200 gate config; a tie closes lr for
  the native regime.
- r113 (B200, RL only, not gated; LOG 11-20 04:34): `d007rl_native_sil256_b200` (`sil_rows` 256),
  `d007rl_native_silc2_b200` (`sil_coef` 2.0) and `d007rl_native_stack3_b200` (lr 12e-6 cosine, `sil_rows` 256,
  mini levels M2, M3, M6), seeds 0 and 1, each the r101 native control with only those settings changed and the
  B200 copy of a staged box arm of r108. Queued 04:21 on 11-20 behind r105; since 09:02 it runs before r105's
  later seeds (LOG 11-20 09:23). Expected against the control at the same seed: sil256 -0.05 to +0.04, silc2 -0.05
  to +0.04, stack3 -0.03 to +0.06 (seed sd 0.019); sil256 and stack3 should cost about 38 % more per step (an
  estimate). Read at matched steps and at the final, with steps and node: +0.02 or more at the final in both pairs
  and ahead at matched steps gets a third seed, box seeds and a place in the next gate config; below the control
  in both pairs both ways is closed and replaced in the box queues; anything between is left to the box's seeds.
  **silc2 seed 0 (LOG 11-20 15:32):** stopped by the pool quota error 422 s before its end and read from its
  partial: -0.025 against the control over 69 shared evals (-0.05 to step 10,000, level after 20,000), last eval
  0.305 against the control's final 0.3033, so a third seed for silc2 is off; not rerun. sil256 seed 0 failed at
  step 4,100 (-0.043 by then) and reruns. **Seeds 1 (LOG 11-20 18:42):** both were stopped at about 3 h by the full pool
  server and read from their partials. silc2 seed 1 is -0.034 over 45 evals to step 22,500 and behind in every block, so silc2 is
  behind at matched steps in both pairs: "between" under the rule's wording (seed 0 was level at the end), not rerun here,
  and a weak candidate for the 11-25 box. sil256 seed 1 is -0.009 over 38 evals to step 19,000, closing to -0.003 after
  step 10,000. sil256 seed 0 reruns first; seed 1 reruns only if seed 0 ends at +0.02 or more over the control's 0.3033
  and is ahead at matched steps (expected: level, -0.02 to +0.01). **sil256 seed 0 (LOG 11-21 02:00):** 0.3033 at
  the final against the control's 0.3033, after 29,892 steps against 34,893 (1.718 against 2.005 steps/s; node5602-c1
  against node5600-c1); -0.001 over 59 shared evals (-0.03 before step 10,000, +0.026 after 20,000). Between, as
  expected: seed 1 does not rerun, and sil256 is left to the box as the strongest r113 arm so far. **stack3 seed 0
  (LOG 11-21 02:39):** 0.2733 at the final against 0.3033, after 31,122 steps against 34,893; -0.008 over 62 shared
  evals (behind at 42). Below both ways, at the bottom of the expected range (-0.03 to +0.06); its parts alone were
  level, so on one seed the stack does not add. **stack3 seed 1 (LOG 11-21 07:39):** 0.2733 against 0.2883 at the
  final after 38,299 steps against 44,862, but +0.0001 over 76 shared evals (+0.005 after step 20,000), so not below
  both ways: stack3 is between, the weakest r113 arm for the box; on seed 1 it is -0.022 against lr 12e-6 alone at
  matched steps.
- r115 (B200, RL only, not gated; LOG 11-21 02:00): `d007rl_native_p32_b200`, r101's native control with half the
  prompts per step (`prompts_per_step` 32, `group_size` 8, so 256 rows a step), `sil_rows` 32 and `eval_every`
  1000; seeds 0 and 1, in place of r105's two seed-2 items. Expected: 1.45x to 1.6x the control's step rate; score
  centre 0.00 at the final, range -0.04 to +0.03 (paired noise about 0.015). Read at the final and at matched train
  time: +0.02 or more at the final in both pairs and ahead at matched time in both gets seed 2 and a place in the next
  gate config and on the box list; below in both pairs both ways is closed; anything between is left there. The
  step rate goes to the speed work in any case. **Closed (LOG 11-21 12:40):** 0.2700 and 0.2567 at the final against
  the control's 0.3033 and 0.2883 (-0.033, -0.032); -0.012 and -0.027 at matched train time (behind at 57 of 64 and
  55 of 58 evals); -0.041 and -0.040 at matched steps (behind at every shared eval). 1.29x the control's step rate
  on seed 1 (1.82x on seed 0, whose control ran on a slow node). Half the rows cut the sampler's time by 26-32 % and
  the update's by 15-31 % on seed 1, so much of a step is fixed; r117 tries twice the rows.
- r117 (B200, RL only, not gated; LOG 11-21 12:40): `d007rl_native_p128_b200`, r101's native control with twice the
  prompts per step (`prompts_per_step` 128, `group_size` 8, so 1,024 rows a step), `sil_rows` 128 and `eval_every`
  250 (every 32,000 prompts, as the control). Seed 0 (29966174, 17:02 to 21:53): final 0.2950 against the control's
  0.3033 (-0.008), +0.007 at matched train time (ahead from the second hour to the fifth), +0.029 at matched steps,
  0.62x the control's step rate; late, the update took 2.0x the control's time for 1.9x the informative groups.
  Between under its rule, so r117 is left there and seed 1 is not run (LOG 11-21 21:58). Expected:
  0.63x to 0.71x the control's step rate, about 103 GB at peak; score centre +0.005 at the final, range -0.03 to
  +0.035. Rule as r115's; its matched-steps reading goes to the speed work in any case.
- U25 (unranked, outside the rules; LOG 11-21 17:59): a doubling search on pretraining size for where a track D model starts to reason, probed without RL. First rung 3x D 007's size (1024 x 24, 332M) on two-phase data (D 007's mix, then the keyword-filtered FineMath, dm_math and FineWeb-Edu at 0.5/0.25/0.25; the fused loss at micro-batch 64), 330 min on a normal B200 (30010825, 21:59 to 03:32 on 11-22: 5.35B tokens, 16.1 a parameter, val loss 2.644 against D 007's 2.86 to 2.90). **The 3x rung says no** (LOG 11-22 03:52): none of set B's 20 samples is working by eye, chain rate 0.05 and step accuracy 0.44; it gained format over 1x (well-formed 0.17 against 0, loops 0.07 against 0.67), not working. The 6-minute smoke run passed (about 4.8B tokens projected for the rung), and D 007's base, probed as the reference, shows no working on the first two sets (LOG 11-21 19:57; `scripts/research/u25_rung.sh`, `scripts/research/probe_reasoning.py`, configs in `calibration/research_r121`); So the search goes up to 6x (1280 x 32, 668M; about 7.9 h for 13.4B tokens on eight H100s at 3x's MFU) on the 11-25 box, which needs data-parallel pretraining and resume (code in, LOG 11-22 04:03; GPU check on one B200, then two, by 11-24), and keyword-filtered FineMath shards fetched on the box (FineMath_kw would repeat about 6.4 times); 12x on the 11-28 or 12-09 box if 6x also says no. Measured on three prompt sets (track D's r1 prompt, the trial-and-error few-shot prompt, GSM8K with worked examples): step accuracy, grounded-chain rate, solve rate, and 20 samples per set read by eye beside D 007's base. "Reasoning appears" is fixed in that LOG entry: on the few-shot set, at least 10 of 20 samples are real working, and a chain rate of 0.5 or more with step accuracy of 0.7 or more, each 0.2 above D 007's base.
- r116 (U24; B200, not gated; LOG 11-21 12:40): pretraining on `mix_0.2.bin` with two appended copies of FineMath
  (`finemath-4plus`, shards 0-2 at revision e92b25a6), keyword-filtered for Countdown and numbers-game items
  (`scripts/keyword_filter.py`); about 31 % FineMath, 14 % dm_math and 55 % FineWeb-Edu by tokens. D 007's recipe
  stopped at its 1,337,982,976 tokens (`d_fmkw2_muon1338m_pre`, seeds 0 and 1), then r101's native control unchanged
  (`d116rl_native_fmkw2_b200`), paired with r101. The control's weights are D 007's own (H200, stopped by time), so
  the data and the pretraining GPU are read together; a B200 control pretraining (r84's config) runs only if the arm
  clears the bar. Expected: centre -0.01, range -0.05 to +0.03; completion length unchanged at about 24 tokens (a
  move to written reasoning is reported whatever the score). Rule: +0.02 or more at the final in both pairs gets
  seed 2 and the B200 control pretraining; below at the final and at matched steps in both is closed; else between.
  Data: ready (normal B200 job 29999304; `mix_0.2_finemath_kw2.bin`, 3.362B tokens; the filter dropped 2,230 of
  314,040 FineMath documents, LOG 11-21 18:16). Three CPU jobs failed before it (CPU nodes do not reach the pool,
  LOG 11-21 14:59). Pretraining: done (normal B200 job 30000622, 18:08 to 19:25, about 38 min a seed; LOG 11-21 19:57). The two RL
  items are queued on `screens_rl_b200`.
- r120 (H200 gate, end to end; LOG 11-21 17:13): `calibration/research_r120/d007_speed_lr12e-6_h200`, r102's
  config with lr 1.2e-5 and nothing else changed (r105 met its rule at lr 12e-6 on the B200). Seeds 0-2 on
  `queue/gate_r120_h200.txt`, preemptable H200 (18,000 s form: `mit_normal_gpu` H200 waits ran to a median of
  214 h in November). Expected: a centre near 0.285 a seed if the B200 gain carries over; three of three at
  0.281 with probability 0.21 to 0.27, a five-seed mean at 0.281 with 0.70 to 0.81. Rule: all three at 0.281
  or more starts a five-seed record attempt (r120's config without `no_save`); a mean of 0.276 or more adds
  seeds 3 and 4 for the five-seed form; under 0.276, no record from lr alone. Paired with r102 by seed.
  **Seeds 0 and 1 (LOG 11-22 11:06):** 0.3100 (43,659 steps) and 0.3017 (46,332), both at 0.281 or more; against
  r102's same seeds +0.040 and -0.008 at the final, +0.036 and +0.014 at matched train time, +0.028 and +0.010 at
  matched steps. Seed 2 (30051396) decides between the three-seed form and seeds 3 and 4.
- r114 (H200 gate in a 6 h form; LOG 11-20 10:26): **Taken off at 15:32 on 11-20 (LOG 11-20 15:32):** r102 seed 0
  ended at 0.270, so three of three at or above 0.281 is 0.3 to 2 %; its items are out of the queue and its
  workers cancelled. As planned: `calibration/research_r114/d007_speed_h200_6h`, r102's config
  with `time_budget_s` 16,440 instead of 18,000 (46 + 274 min, 350 with `run_item.sh`'s 30-minute margin, so a
  seed fits a 6 h `mit_normal_gpu` worker as a screen and, without `no_save`, as a record seed; it cannot be
  preempted there). Seeds 0-2 in `queue/gate_r114_h200.txt`; workers 29777239 to 29777241 (normal, 6 h,
  `RESERVE_MIN` 335), submitted 09:50 on 11-20; planned starts the evening of 11-25 at some readings and none at
  others, behind the B200 workers in my two-GPU cap. Expected from r101's native curves (B200, not gated): 0.262
  at the AWS H100's 1.61 steps/s, 0.277 at 2.0, 0.279 at 2.6; the line is 0.281. Rule: three seeds at or above
  0.281 gives a five-seed record attempt in this form; short of that the gap says what an arm must add, and the
  queued seeds can be swapped for a stronger config (new directory) before a worker starts them.
- `factored_grammar` (commit 1c7883d, off by default, same maths; LOG 11-19 10:12): the native sampler's answer
  constraint on the CPU is 18 to 22 % of the step in r100's timing runs, and on a CPU benchmark its cost is
  building the allowed set of each new grammar state. The option builds that set from parts (tokens without a
  digit cached per state shape, digit tokens read off the remaining numbers): equal sets on every state tested,
  first-pass grammar time 0.30 of the walk's on the login node. **Timed on a B200 at the final weights (r106, LOG
  11-20 04:12): same maths on all 300 steps, no speed.** The constraint timer reads 0.0868 and 0.0875 s a step in
  the two controls and 0.0903 and 0.0865 in the two factored runs (expected: down by a third to 70 %). The option
  only speeds up the first visit to a grammar state, and a trained policy revisits the same states. It does not
  join the settled settings and stays off by default. At the start weights (r106's second pair, LOG 11-20 10:26)
  the timer is lower with the option (0.099 against 0.127 s a step): new states cost time early in a run, but the
  saving fades as the policy settles (about a minute of a run at most), so the option stays off. What the timer's
  0.087 s at the final weights is (a wait for the forward step, or per-row overhead in `mask_rows_cached`) is
  open; r112's smokes did not log it.
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
