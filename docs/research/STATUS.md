# Status

(maintained by the research agent; see AGENT.md)

Updated 2026-11-20 14:24 EST. The run ends 2026-12-19. The agent runs on Opus 5.5.

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
the requeue reset its age (priority 31,298). Seed 2's worker is pending (LOG 11-20 10:26). Read against D's line
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

**Jobs (14:24 EST 11-20).** Since 11:48: 29715907 ended after r105's lr 24e-6 seed 0 (14:16) and 29715909 started at 14:18 on node5801-c1 with r113 sil256 seed 0, so the normal pair is now 29715908 and 29715909; seven normal B200 workers and 29632479 are pending for the eight long items left. As at 11:48: two B200s run on `mit_normal_gpu`, the cap's two normal GPUs: 29715907 (node5800-c1;
started 09:14, 12 h 16 min after submission) ran r111's five smoke runs and has run r105's lr 24e-6 seed 0 since
09:25 (to about 14:20); 29715908 (node5701-c1; started 10:04, 13 h 05 min after submission) ran r106's
start-weights pair and three r112 size smokes and has run r113 silc2 seed 0 since 10:19 (to about 15:15). One H200
runs on `mit_preemptable`: 29450693 (node4802; started 09:40, 31 h 49 min after its release at 01:52 on 11-19)
with r102 seed 0 (to about 15:35; in native RL since about 10:29 at 1.85 steps a second, dev 0.192 at
step 8,000, inside D 007's range there). r102 seed 1's worker 29450694 started 09:54 on node4603, was preempted at 10:04
and requeued; the requeue reset its age (priority 31,298), and its marker in `queue/running/` makes it redo seed
1. Seed 2's worker 29450702 is pending at 40,892. The cluster is full and my fair-share part is 1,222 of 150,000,
so my jobs wait on age (298 an hour; LOG 11-19 21:02 and 11-19 23:29). Eight more normal B200 workers are pending
on queue `screens_rl_b200` (6 h each: the short items, then one long one): 29715909 and 29715910 (submitted 20:58
on 11-19, priority 365,199) and 29755538 to 29755543 (submitted 04:21 on 11-20, priority 363,023); none can start
before a running one ends, whatever their planned starts say. r114's three normal H200 gate workers 29777239 to
29777241 (queue `gate_r114_h200`, submitted 09:50 on 11-20, priority 361,387) come after them in my two-GPU cap;
their planned starts read the evening of 11-25 at some readings and none at others (LOG 11-20 10:26). Pending
workers gain age under the per-user limit too (LOG 11-20 04:12); a requeued one starts again from zero. The B200
workers exclude only the drained `node5500-c1`. **Practice (11-20 04:34):** one pending worker for each queued
long item, submitted when the item is queued, and never chained with `--dependency` (a job waiting on a dependency
or a hold gains no age here). 29632479 (preemptable B200, same queue) started at 11:18 on node5600-c1, ran r112's second block and was
preempted at 11:46, 22 minutes into r113 sil256 seed 0; it is pending again with its age reset (31,226), and I put
that item back at the front of the queue (LOG 11-20 11:48). Below 50,000 the scheduler plans no start (`bf_min_prio_reserve`), so preemptable workers start only when
a GPU is free as it tries them, as r102 seed 0's did at 09:40. The H200 workers exclude node5100 (the pool
filesystem hangs there), node2500 and node4900 (slow) among others. **Maintenance: every node is reserved from
00:00 to 21:00 on 12-15**; a 6 h worker must start by 18:00 on 12-14. The B200 queue holds 9 long items (4 h 51 min each;
r112's smokes have all run): r113 sil256 and stack3 seed 0, r105 lr 12e-6 seed 1, r113's three seed-1 items and
r105's last three, for the eight pending normal workers and 29632479. Held: 29497701/02/05 (B200 preemptable; r97/r99, superseded by r102) and 29445218 (H200; r90). At most 2
normal and 4 preemptable GPUs run at once, the cap of six.

**Settled speed settings for new D arms (LOG 11-17 and 11-18; timed on B200; each is the same maths and off by
default).** Pretraining: `--fused-loss --micro-batch 128` (+21 % tokens per second; 46 min gives about the old
56-min checkpoint). RL: the shared prompt with 65,536-token micro-batches (update 14-18 % shorter; fits the H200),
the native sampler (`rollout_backend: "native"`, 1.40-1.60x faster per RL step) and `reuse_sampler_allowed`
(bit-identical over 300 steps, update -14.5 %). r88 (U18): no effect, parked. Fused pretraining at micro-batch 128
holds 52,866 MiB on a B200 (r107, LOG 11-20 04:12), so it fits an 80 GB H100. `factored_grammar` is not one of
these: timed in r106, same maths, no speed at trained weights.

RULES v3.6 (INBOX 11-16 22:35): a record seed over budget only from idle exit time counts as it is (QUESTIONS when it happens).
**Next decisions:** r102 seed 0's final dev (about 15:35; native RL at 1.85 steps a second on the H200, so 0.275 expected
against the 0.281 line, LOG 11-20 11:48); then, at that reading, which goes first on `mit_normal_gpu` (the B200
workers take the 14:20 and 15:15 slots by priority anyway): r114's three gate workers if seed 0 ends at or above
0.281, otherwise the eight pending B200 workers keep their order;
r102's three seeds and r114's against 0.281, each read with the steps it reaches (then the five-seed record
attempt if the three-seed rule is met); no lr lever for a second gate config: r105's two seed-0 pairs are level at equal steps (lr 12e-6 -0.005, lr
24e-6 +0.002; LOG 11-20 14:24), so seeds 1 stay and the two seed-2 items are replaced by a bold item at the next
full wake (r106's `factored_grammar` gave no speed at the final weights and is not a lever either); where the fixed part of a native step
goes (r112: 640x10 ran 1.10x and 0.77x the control in two blocks, so model size is closed; a 90 s step
rate follows the starting weights, so the timing item compares runs from one checkpoint on one node and logs each
batch's longest completion, LOG 11-20 11:48; the constraint timer's 0.087 s at the final weights is its largest named part, and whether that is
a wait for the forward step or per-row overhead in `mask_rows_cached` is still open; LOG 11-20 10:26), timed
before any code changes; r113's three arms on the B200 (sil256, silc2, stack3; two seeds each; now ahead of r105's
later seeds) against the rule in LOG 11-20 04:34, before the 11-24 planning day if the workers start in time; each
box's standby queue (staged and checked for all three blocks, `docs/research/box_1128.md`) planned again the day
before its block (11-24, 11-27, 12-08), track A arms included if a bold one is ready; each worker's time from
submission to start; and, in the idle wakes before 11-24, bold end-to-end track D arms (U17, U19) for the 11-25
box.

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
  Seeds 1 of both doses stay queued after r113's arms; the two seed-2 items are to be replaced by a bold item. Expected: about +0.02 at 2x (range -0.02 to +0.05) against a paired SE of about 0.005. Read
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
- r114 (H200 gate in a 6 h form; LOG 11-20 10:26): `calibration/research_r114/d007_speed_h200_6h`, r102's config
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
