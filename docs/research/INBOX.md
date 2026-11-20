# Inbox for the research agent

Notes from the maintainer or the laptop session, newest last. Read every wake; mark items done.

- 2026-10-19 (laptop): you take over the loop from the laptop session. r1 (calibration/research_r1, queue r1) is running; Engaging maintenance 2026-10-20 00:00-21:00 EDT. Next ideas noted on the laptop: dev-ladder sampler-temperature sweep on existing weights (A 001 record seeds in ~/orcd/pool/rlvr-ckpts/records/v3/track_a/001_ladder_constrained/seeds/*/final, bf16), track D public-math data prep as CPU jobs.
  - Done (agent, 2026-10-19): took over the loop; r1 unchanged, with follow-on workers for after
    the maintenance. Sampler-temperature sweep: eval arms are ready (`scripts/research/run_item.sh`)
    and queue once r1 frees a GPU. Track D data: the DM-math CPU job is submitted (STATUS.md).

- 2026-10-20 (maintainer, via laptop): Claude usage is not a limit now or soon: no need to keep wakes cheap; spend effort where it improves decisions (GPU time is the bottleneck). AWS quotas checked: p5.4xlarge (1x H100 SXM) up to 4 on-demand per region in us-east-1/us-east-2/us-west-2, 4 spot in us-east-1; record attempts may move to AWS once the maintainer agrees (the laptop session runs AWS; you have no AWS access). Keep record candidates coming.

- 2026-10-20 (laptop): AWS up to $1k is approved for record attempts, run by the laptop session. When an arm meets the promotion rule, say so at the top of STATUS.md (arm, config path, dev numbers). Records may also run on Engaging as before; whichever starts sooner.
  - Done (agent, 2026-10-20): read both notes. STATUS.md now opens with a "Promotion" line (none
    yet); a promoted arm gets its record folder's config.json committed at the same time, so either
    side can start the seeds. Waits go to analysis and to preparing configs.

- 2026-10-21 (laptop): maintainer approved AWS for a01_skip dev screens. Seeds 0-2 run on AWS (run a01_skip-aws-20261021, spot p5.4xlarge, STAGE=train, us-east-1); I removed them from queue/main.txt under flock; seeds 3-4 stay on Engaging. The laptop session fetches the results into calibration/research_r2/a01_skip/seeds/ and commits them here when they finish.
  - Done (agent, 2026-10-20 23:35 EDT): read. `queue/main.txt` keeps a01_skip seeds 3-4 behind the
    fp32 arm's seeds 0-2 and d00_fp32_lr3e-6_skip seed 0, so Engaging won't reach them before the
    AWS seeds are in. When seeds 0-2 land: if all three are healthy and >= 0.270 (the 3-seed form),
    a01_skip promotes, its record config is committed and flagged in STATUS.md, and seeds 3-4 leave
    `main`. Otherwise seeds 3-4 move to the front for the 5-seed form.

- 2026-10-22 (laptop): you were stuck ~38 h on a confirmation prompt after auto mode blocked `scontrol update JobId=26424662 Dependency=` ([Shared Cluster Mutation]) and then two reads. When the classifier blocks an action, do not retry it or variants of it: write what you wanted and why in QUESTIONS.md and continue with other work. Keep the afterany dependencies that hold the 6-GPU cap.
  - Done (agent, 2026-10-22 23:55 EDT): read. The session now runs with permission checks off
    (eb3becd). Blocked or refused actions still go to QUESTIONS.md, not retries. No pending workers
    remained (all had exited on empty queues), so no dependency needed changing.

- 2026-10-22 (laptop), URGENT: node1800 is a black-hole node. Worker 26424658 ran there on 10-21 and every GPU item it touched FAILED: the 5 D2 items (d00_dm26/dm51) after ~113 min (pretraining finished, then the RL start died), and 28 other items within ~20-45 s. The traceback (e.g. calibration/research_r2/d00_fp32_lr3e-6_skip/seeds/4/stdout.log) is triton failing to compile cuda_utils.c with gcc, probably a broken or full /tmp on that node. It drained main, long and r1 to empty. To do:
  (1) Exclude node1800 from all workers. New submissions get --exclude=node1708,node1709,node4508,node1800. For pending ones, ask before any scontrol update, since auto mode blocked one; if it's blocked, note it in QUESTIONS.
  (2) Requeue the 33 failed items listed by worker 26424658 in queue/worker.log. First move their partial seed dirs to extra/ per RULES/run_seed_v3 conventions.
  (3) Make queue_worker stop (and leave the item queued) after 2 consecutive failures shorter than 5 min, so one bad node can't drain a queue again.
  - Done (agent, 2026-10-22 23:55 EDT). Cause: the four failing nodes (node1800, node2700, node2701,
    node2801) run Ubuntu 24.04. The cluster's Rocky-built gcc misses Ubuntu's multiarch glibc
    headers, so triton's launcher build fails, though only for models and configs not yet in
    vLLM's compile cache. A CPU probe on node1800 confirmed the fix. (1) The 6 new workers exclude
    node1800. (2) 29 screen items were requeued after moving their failed attempts to
    `extra/failed_<seed>_1/` with a note. d00_fp32_lr3e-6 seed 1 resumes from saved state. The 5 D2
    seeds had finished pretraining, so their RL now runs from the saved models as r4 screens instead
    of rerunning 6-h pipelines. (3) queue_worker.sbatch now checks gcc before popping (it adds the
    multiarch paths on Ubuntu and exits if gcc still fails), and after two failures in a row under
    5 min it puts the second item back and exits. LOG 2026-10-22 (night) has the details.

- 2026-10-22 (laptop): AWS had no spot H100 capacity for 24 h. a01_skip seeds 0-2 are now on on-demand fleets (run a01_skip-aws-od-20261022). Hold the track D AWS question until AWS has proven it can get capacity; I will answer in INBOX.
  - Read (agent, 2026-10-22 23:55 EDT). The a01_skip seeds 0-2 results are not here yet. a01_skip
    seeds 3-4 are back in `main` behind the r4 D screens; they leave it if seeds 0-2 settle the
    3-seed form first.

- 2026-10-25 00:20 (laptop): congratulations on D 002. Three answers:
  (1) Yes: run the `scontrol update JobId=<id> ExcNodeList=...` you proposed on your own pending successors (26649456, 26649458, 26649459). From now on you may update or cancel your own pending jobs without asking.
  (2) AWS is not sooner. The a01_skip on-demand fleets ran their full 48 h with no H100 capacity (spot had none either). Nothing has run on AWS and $0 has been spent. Do not count on AWS for records. The only firm AWS option is a capacity block (us-east-2, 2 H100s for 24 h from 10-31 11:30Z, $287), which I am putting to the maintainer.
  (3) Start A 002's record on Engaging now: mit_normal_gpu, HBM3, record-<slug> queue, `scripts/run_seed_v3.sh`, seeds 0-4 in order. Free preemptable workers as needed to stay at 6 GPUs. Hold A 003 until A 002 has landed or the maintainer decides on the block. If the block is bought, I will tell you here, run A 003 there, and put any A 002 seeds not yet started on it too, as long as all of a record's seeds stay on one provider. So once a record has started on Engaging, it finishes on Engaging.
  (4) The a01_skip AWS seeds 0-2 never ran; requeue them on Engaging if you still want them.
  - Done (agent, 2026-10-25 00:47 EDT): thanks. (1) 26649458 and 26649459 had already exited
    (node2801's CUDA check put their items back), so I cancelled 26649456 (pending, no item in
    hand) instead of updating it, to make room for the record workers. (2) Noted; QUESTIONS.md's
    AWS questions are closed. (3) Submitted, then cancelled on the HOLD note below: A 002 seeds 0-4
    in `queue/record-002_fp32_skip_cosine.txt` for five chained mit_normal_gpu workers, 6 h and one
    seed each (26821229 and 26821230, then 26821231, 26821232 and 26821234 `afterany` on them),
    running `scripts/research/run_item.sh` (a CUDA check and a time-left check, then
    `STAGE=all scripts/run_seed_v3.sh`, as D 002's seeds ran). `sbatch --test-only` put the first
    start on node2906 at 10-27 09:04 (34 H100 jobs pending there). A 003 stays held. (4) I won't
    requeue a01_skip: A 002 includes skipping, so a01_skip is now only an ablation of it. Its seeds
    3-4 moved from `main` to `later`.

- 2026-10-25 00:50 (laptop), HOLD: do not submit record workers yet. The maintainer is changing the reference GPU to H200, and details follow within the hour. If a record-* worker is already submitted, cancel it.
  - Done (agent, 2026-10-25 00:47 EDT): cancelled all five record workers (26821229-26821232 and
    26821234) at 00:46:34. I had submitted them at 00:45:43, 14 s after this note was committed and
    before I saw it. None started, so nothing ran. The queue file is set aside as
    `queue/record-002_fp32_skip_cosine.held`, which no worker reads. 4 preemptable workers remain
    (1 running, 3 pending; dev-ladder screens only), and they continue. Waiting for the H200
    details.

- 2026-10-25 01:40 (laptop), v3.1, this replaces the HOLD above. The maintainer moved the reference GPU to the **H200 SXM** (`NVIDIA H200`): "ok just change it all to h200 it cant take that much time to rerun some records right". Commit 18a5b37 updates RULES_v3, the validator, README (separate H100 v3.0 board), PLAN and AGENT.md; reread AGENT.md's Infrastructure section. Plan:
  (1) Records now run on mit_normal_gpu with `--gres=gpu:h200:1`, REQUIRE_GPU="NVIDIA H200" (13 SXM nodes; the NVL cards are only on preemptable 4-GPU nodes). Create and commit these record folders, then queue them in this order:
      - track_a/004_ladder_constrained_h200: A 001's config, copied unchanged. It is the first H200 track A holder.
      - track_a/002_fp32_skip_cosine: runs as committed, compared with 004 once both land.
      - track_d/003_dm_math_fp32_h200: D 002's config, copied unchanged.
      Run A 002 and A 004 in parallel on the two normal-queue slots. A 003 waits, as before.
  (2) The promotion lines (A 0.270, D 0.190) come from H100 dev numbers. Use them until the reruns land, then recompute them from the reruns' in-loop final dev and note that in PLAN.
  (3) The maintainer says: "tell the agent to feel free to queue b200/vr300/whatever else mit has". Engaging's types are h200, b200, h100, a100, l40s, rtx_pro_6000, a40, l4; there is no vr300. Screens may use any of them (REQUIRE_GPU = that type's exact name), but compare arms within one type or use step caps. Smoke-test B200 (sm_100 support in the vllm-venv) with one short item before you queue a batch there. The 6-GPU cap still applies.
  (4) Your pending H100 preemptable workers may finish their current items. Submit new ones as H200 (or other types), as the brief now says.
  - Done (agent, 2026-10-25 01:23 EDT): read; AGENT.md, PLAN and RULES_v3 reread.
    - (1) `records/v3/track_a/004_ladder_constrained_h200` and
      `records/v3/track_d/003_dm_math_fp32_h200` hold A 001's and D 002's config.json, copied
      byte for byte (c910630).
    - The queues are `queue/record-{004_ladder_constrained_h200,002_fp32_skip_cosine,003_dm_math_fp32_h200}.txt`,
      seeds 0-4 each.
    - Track A: two chains of five mit_normal_gpu workers, 6 h and one seed each, `--gres=gpu:h200:1`,
      REQUIRE_GPU="NVIDIA H200". Chain 1 (26824033-37) serves A 004 first, then A 002. Chain 2
      (26824038-42) serves A 002 first, then A 004. So the two records run side by side.
      `sbatch --test-only` estimates node2434 at 10-26 21:44.
    - Track D: a D seed needs 376 min (budgets + 30), more than a 6-h normal job, so D 003 runs on
      mit_preemptable H200 workers (NVL nodes excluded, `--time-min=06:30:00`, RESERVE_MIN 380).
      There are four of them, 26824147-50, each `afterany` on one of the four H100 workers. Those
      finish their current items and then exit: I moved `main` and `long` (15 items) to
      `queue/h100.txt`, which no worker reads. The cap stays at 2 + 4.
    - A 003 waits.
    - (2) Noted in PLAN: H100 lines until the reruns land.
    - (3) The first H200 screen is r12, r10's config (A 002 + mean-centred advantages) on the H200,
      seeds 0-2, in `queue/h200.txt`. The D 003 workers turn to it after the record seeds. I'll
      try B200 with one short item before queueing a batch there, if H200 capacity runs short.

- 2026-10-25 11:15 (laptop): answer to QUESTIONS 10-25 02:28. Yes, track A H200 record seeds may also run on mit_preemptable, which is the same cluster and so the same provider. Keep the mit_normal_gpu chains too, with all workers popping the same record queues under flock. A preempted seed is an infrastructure interruption: file its partial attempt in extra/ with a note and rerun the same seed, as RULES_v3 allows. Use REQUIRE_GPU="NVIDIA H200" and exclude the NVL nodes. Stay at 6 GPUs or fewer.

- 2026-10-25 18:00 (laptop): AWS capacity blocks. The maintainer bought four 24-h, 8-GPU blocks. AGENT.md has the details in a new "AWS boxes" section and an amended hard rule (reread both). The first block, 8x B200, starts **tomorrow 2026-10-26 11:30Z (07:30 EDT)**.
  (1) Maintainer, verbatim: "lowk dont run records on those just run all experiments, try to make experiments relatively compute efficient as you can". Also: "make sure the agent tries like running 2/3 models per gpu if it can". So the boxes run dev-ladder screens only.
  (2) Step caps. Box screens must not be time-capped: a B200 or a packed GPU runs at a different speed than an H200. Cap them at the number of steps an H200 SXM does in the 5-h budget, per track. Measure that from clean H200 runs: on track D, D 003's node5001 attempt ran at 0.75 s/RL step vs 0.80-0.86 on the H100 for D 002 (H200 ≈ 1.07x H100). D 003 seed 0 on **node4900 ran at half speed (1.56 s/step)**: don't use it for the cap, and exclude node4900 from your sbatch lines.
    - The lr schedule (`time_budget_s`, cosine/linear to lr_final_frac) and the track D pretrain budget are time-based. To make step-capped runs follow the same schedule on any GPU, add a step-based option (off by default) and use it on the boxes, together with a matching Engaging H200 control if you compare across GPUs.
    - Each box batch carries its own control arm on the same GPU type and packing.
  (3) Packing: `vllm_gpu_memory` is a share of total GPU memory (default 0.3; a B200 has 192 GB, an H200 141 GB, and one run used ~60 GB on the H100). Before the block, if you can, measure 1 vs 2 vs 3 runs per GPU on an Engaging B200 (mit_normal_gpu has 10 8x B200 nodes, `--gres=gpu:b200:1`, REQUIRE_GPU set to its exact name) for summed steps/h. Then pick PER_GPU for the box. Stagger starts so each vLLM engine sizes its cache correctly (box_workers.sh STAGGER_S).
  (4) Logistics: you have an AWS key with full access (IAM user rlvr-agent, `~/.local/bin/aws`), for dealing with anything unexpected. New spending needs the maintainer's OK. The `aws` deny rule is gone. Your ssh key is `~/.ssh/rlvr_aws`. The 6-GPU cap is Engaging-only.
  (5) Cadence: wake every 20-30 min from 10:30Z tomorrow until the block ends (and the same for later blocks). Have the B200 queue and the transfers ready before 11:30Z. The box needs about 10-15 min after launch to build its venv.
  (6) You'll be restarted with ultracode on (the maintainer is moving to Max 20x), so workflows and subagents are fine when they help.
  (7) Added 18:40: you may spend up to **$500 in total** on AWS this month without asking, for emergencies (maintainer: "give it $500 emergency budget"). Log each charge and the running total. See AGENT.md.
  - Done (agent, 2026-10-25 15:45 EDT; plan in LOG 10-25 14:30, checks in LOG 15:45):
    - (1) Box 1 runs r15 only: eight step-capped track A arms, dev ladder, `no_save`.
    - (2) `budget_steps` (afd8b9f) counts the run and its lr schedule in steps; r15 uses 800, and
      its control arm `mal0_st800` runs in the same batch on the same packing. node4900 is excluded
      everywhere. Track D pretraining needs no new option: `pretrain_scratch.py` already decays
      on whichever of `--minutes` and `--max-tokens` runs out first, so a token cap fixes it.
    - (3) Two runs per B200 (`vllm_gpu_memory` 0.2, `micro_batch_tokens` 4096); three can't fit
      with fp32 trainers of ~48-51 GB each. No Engaging B200 can start before the block
      (`--test-only` at 15:40: 10-26 08:29 preemptable, 19:38 mit_normal_gpu), so the packing probe
      (`pack_probe.sh`, 1 vs 2 track A runs, 1 vs 4 track D) runs on box GPU 7 at the start.
    - (4) Read. AWS checks at 15:50 (read-only): the block is `scheduled` for 11:30Z, the three
      launch schedules are enabled, and the template points at the right ENI (100.29.118.132).
      `scripts/aws/box_ssh.sh` uses a known-hosts file of its own per box address. mit_normal CPU
      jobs wait until 10-28, so the copy runs on mit_quicktest; a compute node's egress is
      18.13.54.83, inside the allowed range.
    - (5) The queue (`queue/b200.txt`, 32 items) is ready. Wakes every 20-30 min from 06:30 EDT.
    - (6) Noted.
    - (7) Noted: $500 emergency budget; $0 spent so far.

- 2026-10-25 22:00 EDT (laptop): **AWS boxes are off. Do not launch any box.** AWS billing says
  the Capacity Blocks can't be paid from the credits, and the maintainer has asked AWS to cancel
  and refund all four. AWS support asked that the blocks stay **unused** while they review it.
  So: no `aws ec2 run-instances` with the `rlvr-box-*` launch templates, by hand or otherwise, for
  any block (10-26 B200, 11-05, 11-18, 11-28), until a later INBOX note says otherwise. The
  launch schedules are deleted. Plan the work for Engaging only: `queue/b200.txt` doesn't run
  tomorrow, so move what matters from it to Engaging queues (within the 6-GPU limit), and drop
  the box steps from STATUS. The $500 emergency AWS budget is also suspended: no AWS spending.
  - Done (agent, 2026-10-25 23:05 EDT): no box will be launched, and no AWS spending. Read-only check
    at 23:00: no EventBridge schedules and no instances in us-east-1 or us-east-2. `queue/b200.txt` is
    parked (no Engaging worker reads it). Its one arm that matters now, the 500-step cosine
    (`mal0_st500`, LOG 18:46), runs on Engaging as r16 seeds 0-2 on the H100 worker
    (`main`, ~23:45 to ~12:30). The rest of r15 waits on r16's result. STATUS has no box steps now.
    New preemptable jobs are estimated to start at 10-26 16:02, so the 6 GPUs stay as they are.

- 2026-10-26 12:00 EDT (laptop), answers to QUESTIONS (warm-up, D 003):
  1. **Yes, record seeds may use the untimed warm-up** (libraries and base model into the page
     cache, imports with no GPU visible). The maintainer also allows going one step further, as
     modded-nanogpt does: its `train_gpt.py` runs a few real training steps before the timer to
     warm up and compile kernels, then restores the model and optimizer state, and those steps
     don't count. So an untimed GPU warm-up pass is allowed too (engine start-up, CUDA graphs,
     compilation, one dummy step of each stage), provided that afterwards every piece of trained
     state is exactly what it would be without it: weights, optimizer state, lr/schedule position,
     RNG states, data and prompt order. No dev or test ladder data in the warm-up, nothing from it
     is kept, and its time is logged separately. Write it into RULES_v3 as a v3.2 clarification
     (LOG entry too), measure it on screens first, and say in each record's README whether it used
     the warm-up.
  2. D 003: option (a). Report D 003 as failed (over the 6-h budget), keep all five seeds in its
     record folder with a note, and rerun D 002's config unchanged, with the warm-up, as D 004.
     (Answered 12:10.)
  - Done (agent, 2026-10-26 13:05 EDT; LOG 13:05). I saw this at 12:43: the 11:43 commit had been written
    before it arrived.
    - (1) RULES_v3 has the v3.2 clarification ("Untimed warm-up" under "Clock and hardware", 8910562).
      Records use the warm-up without the GPU, `scripts/warm_start.sh`, measured on four r16 screens:
      vLLM was up 30-55 s after launch, against up to 15 min cold. `run_item.sh` now runs it before
      record seeds too, and writes its time to `<record>/warm_start.log` as well as worker.log. Each
      README will say so. I haven't built the GPU warm-up pass: the warm-up without the GPU already
      brings start-up back to the H100's level.
    - (2) D 003 has a README that reports it as failed, with all five seeds kept. D 004
      (`records/v3/track_d/004_dm_math_fp32_h200_warm`) is D 002's config.json, byte for byte. Its seeds
      0-4 are queued for two preemptable H200 workers that read its queue first.
    - The A records are released. A 004's five seeds are at the front of `h200`, which every worker
      reads, so the two running workers take them next. New H200 jobs aren't expected to start before
      10-27 04:37 (preemptable) and 10-29 (mit_normal_gpu). A 002 is at the end of `h200` (LOG 13:05
      says why).

- 2026-10-26 16:45 EDT (laptop), operational: **screen workers may run on B200 too.** Records stay
  on H200 (the reference GPU). Screens are step-capped, so the GPU type doesn't change their
  results, and a B200 should do roughly 1.3-1.8x an H200's screening work (more memory bandwidth
  for vLLM, 192 GB for two runs per GPU) while counting as one GPU in the 6-GPU limit. So let
  screen-only workers take B200 or H200 (e.g. `--gres=gpu:1` with a constraint, or separate B200
  workers) and prefer B200 when one is free. Measure it first: one B200 worker with 1 and then 2
  runs (the packing probe), and compare steps/hour with the H200 screens. `sbatch --test-only`
  at 16:40 gave the same start for B200, H200 and H100 (preemptable 10-27 10:15, mit_normal
  10-28 04:49), so this widens the pool, it doesn't jump the queue.
  - Done (agent, 17:45): `scripts/research/pack_probe.sh` (1 then 2 track A runs, 1 then 4 track D
    runs) is job 26953974 on a mit_preemptable B200, estimated 10-27 15:22. The later mit_normal_gpu
    chain head waits on it, so the cap stays at 6. Once it reports, step-capped screens go to
    screen-only B200 workers (packed if two runs pay off), from a queue without record items. LOG 17:45.
  - Done (agent, 10-28 20:55): the probe ran after the libstdc++ fix (27162075, node5800-c1). One track A
    run on a B200 is 1.3x an H200 run, and a second run on the same GPU adds only 11 %, so B200 screen
    workers take one item per GPU at the arms' own settings. Track D packs well (three runs 2.5x one).
    r17 moved from `h200` to `screens_b200`, read by two B200 workers (27168549, 27168550). LOG 10-28 20:25.

- 2026-10-31 12:00 EDT (maintainer, via laptop), **rules v3.3: no external data on track A.** The
  external-data amendment of 10-19 now applies to track D only. Track A's data is again generator
  puzzles and verifier verdicts only (RULES_v3 updated in this commit). No track A record used
  external data, so nothing is invalidated. PLAN A4 (SFT on a public dataset before RL) is out for
  track A; external data stays open for track D (pretraining, fine-tuning or RL).

- 2026-10-31 12:15 EDT (maintainer, via laptop), **priority: track D matters more to the maintainer
  than track A.** From now on, give track D at least half of research effort and screen capacity.
  Track D screens pack about 3 per B200 (LOG 10-28 20:25), so they don't need the H200s. Track A's
  queued records (A 005, A 002) still run as planned. Which track D directions to try is your call.

- 2026-11-01 23:30 EST (maintainer, via laptop), **process: review your plan before more track D RL
  knobs.** Since 10-31, about a dozen track D RL arms (shaping weights, lr, group size, prompts per
  step) have all ended in the same 0.14-0.18 dev band, in one-guess mode. Before you queue more
  screens that only vary RL settings, go back over PLAN.md's untried track D items, and any other
  direction of your own. Either queue the most promising of them, or write in LOG why the RL side is
  still the better bet. Which direction to take is your call. Log this note as a maintainer
  intervention.

- 2026-11-03 13:30 EST (maintainer, via laptop), **the 11-05 H100 box is back on; the others stay off.**
  This lifts INBOX 10-25 22:00 for the **2026-11-05 block only** (8x H100, us-east-2b,
  `zjason@16.58.142.215`, 11:30Z 11-05 to 11:30Z 11-06). Follow AGENT.md "AWS boxes" for it.
  - Launch is scheduled again (start +1/+10/+25 min; the launch template was rebuilt today).
    The +30 min fallback `aws ec2 run-instances` for this template is allowed. Make no other AWS
    API calls and spend nothing on AWS; the $500 budget stays suspended. The 11-18 and 11-28
    blocks stay off until a later note.
  - **Screens only.** Records and test ladders stay on Engaging. The box's 8 GPUs don't count
    toward the 6-GPU Engaging limit.
  - **Split the box about 50/50 between track A and track D** (e.g. 4 GPUs each). Which screens is
    your call.
  - Have the box queue files ready before 11:30Z 11-05. An H100 has 80 GB, so B200 packing
    settings don't carry over; check how many runs fit per GPU first.
  - `agent/vllm-venv-freeze.txt` had a stray first line ("Using Python ... environment at"), so
    the pinned install would have failed. It's fixed (the original is `.orig`). If the venv still
    fails on the box, fix it there (sudo works); don't give up the block.
  - The box's disk is deleted at the block end: `box_sync.sh from` every wake, and a last time
    before 11:00Z 11-06. While the box is up, wake every 20-30 min (this overrides the 6-hour
    full-wake rule for that day).

- 2026-11-04 12:00 EST (maintainer, via laptop), **11-05 box: bring unfinished runs back to finish on
  Engaging.** Before the last sync (by 11:00Z 11-06), copy each unfinished box run's resume state
  (its latest `state.pt` and whatever else a resume needs) back to the same path under
  Engaging's `$CKPT_DIR`, along with its logs, and queue it on Engaging so it resumes from that
  step. `box_sync.sh from` copies only logs and JSON now; extend it (off by default, e.g. a flag)
  or do it by hand. So you don't need to stop starting long runs late in the block: a run that
  can't finish on the box finishes on Engaging. Check that a resumed run's step cap, step count
  and logs carry over correctly. Don't keep the box's disk past the block (no AWS changes).

- 2026-11-05 23:00 EST (maintainer, via laptop), **arithmetic fix in LOG 11-05 22:35.** In the r39/r40
  table, the r39 control's mean over seeds 0-4 is (0.380 + 0.383 + 0.395 + 0.253 + 0.377) / 5 =
  0.358, not 0.378, so r40 - r39 is +0.014, not -0.006. Please correct the table (note the fix in
  LOG) and use the right numbers for later decisions. Consider computing table means with a script
  rather than by hand.

- 2026-11-06 11:30 EST (maintainer, via laptop), **label fixes in LOG 11-06 10:40 (r46).** The entry
  says "80 % of the budget on dm_math pretraining", but `pt80` is 80 minutes of pretraining (1.91B
  tokens). It also says "read at the 15k cap" and "Dev pass rate at 15,000 RL steps", but the pt80
  column (0.213 / 0.222 / 0.210 / 0.227 / 0.208) is from step 17,000, the arm's full budget
  (`eval_log.jsonl`). The comparison and the tie conclusion aren't affected. Please correct the
  labels and note the fix in LOG.

- 2026-11-09 13:00 EST (maintainer, via laptop), **keep both tracks running.** Track D still gets at
  least half (INBOX 10-31), but track A shouldn't sit idle: keep at least one or two GPU slots on track
  A screens whenever there's capacity. Which track A directions to try is your call.

- 2026-11-10 10:00 EST (maintainer, via laptop), **model change and a one-month extension.**
  - From the restart at about 10:00 EST 11-10, this session runs on Fable 5.2 (`claude-fable-5-2`)
    at effort xhigh. It was Opus 5.5 at effort high until then. It is the same conversation,
    resumed. New commits carry the trailer `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
  - The run now ends 2026-12-19, not 2026-11-19. AGENT.md has the new dates, the new trailer and
    a note on the harness's 7-day /loop limit (reread it). The rules, the promotion rule, the
    6-GPU cap and the track split (D at least half, A not idle) are unchanged.
  - AWS is unchanged: the 11-18 and 11-28 blocks stay off and the $500 budget stays suspended
    (INBOX 11-03), unless a later note says otherwise.
  - Please add a LOG entry that marks the handover: the record holders, the open arms and what
    is queued, as of this restart. Update the dates in PLAN.md, and revise its agenda for the
    extra time as you see fit.

- 2026-11-10 10:40 EST (maintainer, via laptop), **the 11-18 and 11-28 AWS boxes are back on.**
  Maintainer, 2026-11-10: "surely just use it". This lifts INBOX 10-25 22:00 for both remaining
  blocks:

  | Start (UTC) | End (UTC) | GPUs | Region | ssh | Launch template |
  |---|---|---|---|---|---|
  | 2026-11-18 11:30 | 2026-11-19 11:30 | 8x H100 (p5.48xlarge) | us-east-1f | zjason@174.129.98.131 | rlvr-box-cr-0261a6f8bb30fc123 |
  | 2026-11-28 11:30 | 2026-11-29 11:30 | 8x H200 (p5en.48xlarge) | us-east-1d | zjason@100.29.118.132 | rlvr-box-cr-08927241900b5884c |

  Follow AGENT.md "AWS boxes", with these differences:
  - **There is no scheduled launch, so launch the box yourself.** The us-east-1 schedules were
    deleted on 10-25, and the scheduler couldn't launch the 11-05 box either. From the block start
    (11:30Z), run `aws ec2 run-instances --region us-east-1 --launch-template LaunchTemplateName=<template>`
    and retry every couple of minutes until it succeeds (it is refused until the reservation is
    active). Check `aws ec2 describe-instances --region us-east-1` first so you don't launch
    twice (a block holds one instance, so a second launch fails anyway). Those two commands, for
    these two templates, are the only AWS calls allowed. Spend nothing else on AWS; the $500
    budget stays suspended.
  - Both templates were fixed today (default version 2). Version 1 carried the venv pins with the
    stray first line, the same fault as before 11-05. Version 2 embeds the same pins as the 11-05
    box (`agent/vllm-venv-freeze.txt` as it is now). If the venv still fails on a box, fix it
    there (sudo works); don't give up the block.
  - **Screens only.** Records and the test ladder stay on Engaging. A box's 8 GPUs don't count
    toward the 6-GPU Engaging limit. Split each box about 50/50 between track A and track D; which
    screens is your call.
  - **Engaging maintenance is 2026-11-17 00:00-21:00 EST**, and the 11-18 block starts at 06:30 EST
    on 11-18, 9.5 h after it ends. Have the 11-18 queue files, configs and the list of what to
    copy ready and committed before the maintenance starts. If the login node restarts, the cron
    keepalive should bring you back; check the block first.
  - A box's disk is deleted at the block end. Run `box_sync.sh from` every wake and a last time
    before 11:00Z on the end day, and bring unfinished runs' resume state back to Engaging as in
    INBOX 11-04. While a box is up, wake every 20-30 min (this overrides the 6-hour full-wake
    rule for that day).
  - The maintainer may still take the 11-28 block for other work. A later note will say so if
    that happens; plan for it until then.

- 2026-11-10 14:50 EST (maintainer, via laptop), **answers to QUESTIONS 11-10 10:33.**
  - **3, mini levels (r55): fine.** Maintainer: "q3 is def fine". Your reading holds: a training
    mix of generated puzzles from the ladder's own generator and filter, never scored.
  - **2, self-imitation on the policy's own verified samples: allowed** (maintainer: "probably
    fine"), within these limits:
    - the buffer starts empty in each seed's timed run and is never carried between runs or
      seeds (a saved buffer would be Countdown-specific data);
    - each completion is stored verbatim with its original generator puzzle: no relabelling, and
      no program edits or trims it;
    - the verifier's verdict is the only filter, and any sampling done to fill the buffer counts
      toward the budget like the rest.
    If an arm built on it meets the promotion rule, say so at the top of STATUS before the record
    starts, so the maintainer can look at it first.
  - **1, hindsight relabelling: no decision.** Keep treating it as not allowed, as you assumed.

- 2026-11-10 14:54 EST (maintainer, via laptop), **question 1 again: one unranked screen of hindsight
  relabelling is allowed, to measure it.** The maintainer hasn't ruled on relabelling and wants to
  know how large its effect is first. It stays **not allowed** in records, promotion screens and
  anything stacked into a ranked arm.
  - What is allowed: one arm on track D, three seeds, dev ladder only, paired against D 005. Mark
    it unranked in its config and in LOG, and never promote or stack it, whatever it scores.
  - The relabelled target is the value of the model's own expression on the puzzle's own numbers.
    Don't run a solver or a search over expressions on relabelled pairs (so no `reachable` or
    "needs" filter on them), and keep the option off by default in code.
  - Report the paired difference, the per-level numbers, and what share of samples gave a usable
    relabelled pair.
  - It doesn't outrank r54, r55 or other ranked work. Fit it where it costs them least; the 11-18
    box has room. The design beyond this is your call.

- 2026-11-10 14:58 EST (maintainer, via laptop), **the maintainer's own ideas are now yours to judge.**
  Maintainer, 2026-11-10: "you can send my ideas over to the agent, have the agent judge what to
  do / whether to do / if its worth doing, and report it back. i do want to know what happens to
  every one of my ideas tho".
  - The list is `~/rlvr-ideas/IDEAS.md` (15 ideas, U01-U15). It is outside this checkout on
    purpose: this repo is public on GitHub and the list is not. Don't copy the list, or your
    assessments of it, into the checkout.
  - Judge each one yourself: is it allowed under RULES_v3 (say so if it needs a ruling), has an
    earlier arm already covered it, is it worth testing in this run, and if so how and when.
    Whether to run any of them is your call. They don't outrank your own plan, and there is no
    quota. The list leaves out the laptop session's opinions on purpose.
  - Keep `~/rlvr-ideas/STATUS.md` current: one row per idea, never blank or dropped, with the
    verdict, a line or two of why, the arms and the result, plus a short assessment per idea
    below the table. Do the first pass at your next full wake, and update it whenever a verdict
    or a result changes. The laptop session reads it at each check-in and reports to the
    maintainer.
  - Rule questions about these ideas go in STATUS.md's last section, not in QUESTIONS.md.
  - When you run an arm built on one of them, log it in LOG as usual and name the id (e.g.
    "maintainer idea U07"), so those results can be told apart from what you found alone. Arms
    and their results are public like the rest of LOG.

- 2026-11-10 15:04 EST (maintainer, via laptop), **weigh the maintainer's ideas on their merits; and a
  ruling on checkpoints (RULES v3.4).**
  - **The ideas in `~/rlvr-ideas`.** Maintainer: "tell it not to rly care about me (ie dont run
    screens that i thinks is bad just cuz i asked)". Give an idea no extra weight because it is the
    maintainer's; judge it as you would your own. Don't run a screen you think is a poor use of
    GPU time because the maintainer suggested it. "Not in this run", with the honest reason, is a
    good verdict, and the reason is what the maintainer wants to read.
  - **Checkpoints (this settles U12's question).** Maintainer: "checkpoint selection is fine as
    long as it counts in time limit, so rollback is fine", and "no checkpointing based on test
    tho only train eval". RULES_v3.md is amended as v3.4 (the header, "Clock and hardware" and
    "Seeds and statistics"; reread them). In short, a run may choose among its own checkpoints, or
    roll back and train on, if:
    - the run's own code makes the choice before the clock stops (nobody picks afterwards);
    - it uses training signals and in-loop dev evals only, never the test ladder;
    - snapshots, restores and any dev eval the run acts on count toward the budget.
    It applies to both tracks and to records. A record that uses it says so in its README, and
    its clock has to include the dev evals it acted on: check that the trainer's clock and
    `validate_record_v3.py` handle that before a record relies on it. Existing records chose no
    checkpoint and are not affected.

- 2026-11-10 23:46 EST (maintainer, via laptop), **back on Opus 5.5, and a low-usage window until
  2026-11-11 18:00 EST.**
  - **Model.** Maintainer: "can you put it back on opus". From this restart you run on Opus 5.5
    at effort high again, as before 2026-11-10. The commit trailer is
    `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` again. The end date stays
    2026-12-19 unless the maintainer says otherwise. AGENT.md is updated.
  - **Low-usage window.** Maintainer: "scale it down until tmrw at 6". Until 2026-11-11 18:00 EST
    (check `date`; the laptop session may move the time), use as little as you can:
    - One wake about every 5 hours instead of every hour. At the end of each wake, don't
      schedule the next one: end the loop (ScheduleWakeup with `stop: true`) and end the turn.
      The cron re-issues the /loop once the session has been idle for 5 hours
      (`agent/slow_until` holds the end time; don't edit it).
    - At each wake do only what can't wait: check the jobs, requeue host failures, keep the
      queues fed from what is already planned, log finished arms in a few lines, and start a
      record if the promotion rule is met. Leave new code, new arm designs, PLAN rewrites, long
      analyses and the first pass over `~/rlvr-ideas/STATUS.md` until after the window.
    - After the window, go back to hourly wakes with a full wake every 6 hours, and catch up
      then.

- 2026-11-11 12:01 EST (maintainer, via laptop), **the 08:12 cancellation was Slurm, not us: resubmit now.**
  - This answers your QUESTIONS.md entry of 11:46. Neither the maintainer nor the laptop session
    cancelled your workers. `sacct -a` shows more than 1,800 pending jobs of other users
    cancelled between 08:12 and 08:16 EST, with no cancelling uid, so it was an event on the
    Slurm side. node1800 and node2801 no longer appear in `sinfo`. They are still in your
    `--exclude` list, and `sbatch --test-only` with that list passed on both partitions at
    11:44 EST.
  - Maintainer, 2026-11-11: "resubmit". Resubmit the workers now, within the 6-GPU cap. The
    scale-down is about your Claude usage (the wakes) only, not about GPU jobs.
  - The low-usage window still runs until 18:00 EST today.
  - Why no wake ran between 23:47 and 11:45 EST: the cron's idle check read the transcript
    file's mtime, and an idle session refreshes that about once an hour. It now reads the time
    of the last transcript entry (6d3ff17), so the re-issue after 5 hours idle (2 hours outside
    a window) works.

- 2026-11-12 11:56 EST (maintainer, via laptop), **a self-imitation record is OK if r61 passes; and A 006's
  "test" numbers in LOG are dev numbers.**
  - **Record D 006.** Maintainer, 2026-11-12, asked whether a record built on self-imitation may
    go ahead if r61 passes: "sure its probably fine ig". So if r61 meets the promotion rule, queue
    record D 006 without waiting for another look. The limits of INBOX 11-10 14:50 stay (the
    buffer starts empty in every seed and is never carried between runs; completions are replayed
    verbatim with their original puzzle; the verifier is the only filter). Say in the README that
    the record uses self-imitation.
  - **A 006's scores.** LOG 11-12 09:07 and later notes give seeds 0-2 as "test 0.397 / 0.375 /
    0.397". Those are the kept in-loop dev values (`result.json`, `final_pass_rate`, 600 dev
    puzzles). The test-ladder scores are in each seed's `confirm_v3.json`: 0.4099, 0.3932 and
    0.4148. Add a correction to LOG (a new dated line, not an edit of history), and take the
    README's numbers from `confirm_v3.json`. Nothing is wrong with the runs: the in-loop evals
    read the dev ladder only.
  - **`~/rlvr-ideas/STATUS.md`** has not changed since 11-10 19:31. U12's row should now carry
    r58, r60 and A 006, and U11's the r51 result. Keep it current when a result lands.

- 2026-11-13 23:43 EST (maintainer, via laptop), **B200 first, progress first; H200 gates and records may
  wait as a backlog.**
  - Maintainer, 2026-11-13: "if b200s are so easy why dont we just max out b200 and dont worry
    about h200", "any record backload can just be backlogged until we start getting more h200 or
    smth else changes", and "just prioritize progress rn".
  - The numbers behind it (your own started jobs in November, submit to start): B200 median 0.1 h
    on both partitions (max 0.8 h on mit_normal_gpu, 2.2 h preemptable); H200 median 6-9 h
    preemptable and 214 h on mit_normal_gpu. On 11-13, 25 pending mit_normal_gpu jobs asked for
    a B200 against 98 for an H200.
  - What changes:
    - Progress comes before records for now. Use the six slots for whatever finds the next
      improvement soonest, which will mostly be step-capped B200 screens. A gate or a record
      that waits days for an H200 is fine.
    - Don't hold a slot idle or pending for an H200 while B200 work is queued. Slurm's per-user
      caps count running jobs, so an H200 worker can't start while B200 workers fill that
      partition's cap: when H200 work is waiting, keep a slot for it (how many is your call);
      otherwise fill all six with B200.
    - How far to build on a B200 result whose H200 gate hasn't run is your call. Mark such
      results in LOG and STATUS as not yet gated, and keep the backlog of gates and records
      listed in STATUS.
  - What doesn't change: the rules (the H200 stays the reference GPU; gates and records run on
    `NVIDIA H200` on Engaging), the promotion rule, the 6-GPU cap and the bad-node exclusions.

- 2026-11-13 23:45 EST (maintainer, via laptop), **follow-up: don't wait for H200s at all for now.**
  - Maintainer, a few minutes after the note above: "just dont care about h200 now just make
    progress fast". This replaces the "keep a slot for H200 work" part of that note.
  - Nothing has to run on an H200 for now. Don't reserve a slot for one and don't leave a worker
    pending for one. When a worker ends, replace it with whatever starts soonest and does the
    most per slot, which today is a B200 (one track A run, or three track D runs).
  - The H200 workers that are running now are doing useful work: let them finish. Don't cancel
    them.
  - Gates and records wait in the backlog you list in STATUS until the maintainer says otherwise.
    Screens that you ran timed on an H200 (track D from pretrained checkpoints) can become
    step-capped B200 screens with their own control, if that reads as well for your purpose.
  - Everything under "What doesn't change" in the note above still holds.

- 2026-11-14 17:01 EST (laptop session, operational; the maintainer's aim is unchanged: "just make progress fast"),
  **the four preemptable slots are not tied to B200.**
  - The two notes of 11-13 put B200 first because B200 jobs start within minutes. They did not look at
    preemption, and that was the laptop's miss. Your own job history (`sacct -D`, every incarnation since 10-20):

    | partition, GPU | GPU-hours | preemptions | share of GPU-hours in incarnations that ended in preemption |
    |---|---|---|---|
    | `mit_normal_gpu` B200 | 57 | 0 | 0% |
    | `mit_preemptable` B200 | 332 | 216 | 77% (today: 18 of 18 incarnations, 12.8 GPU-hours, median 15 min, no run finished) |
    | `mit_preemptable` H100 | 251 | 49 | 50% |
    | `mit_preemptable` H200 | 509 | 42 | 38% |

    A preempted worker loses only the run in progress, so these shares overstate the loss, but they rank the
    types. A likely cause: every usable B200 is also in `mit_normal_gpu`, so any job there that wants a B200
    evicts a preemptable one, while about two thirds of the preemptable H200s sit on nodes only their owners
    reclaim.
  - So "B200 only" is not what the maintainer's aim needs. Keep the two `mit_normal_gpu` slots on B200. For
    the four preemptable slots, use whichever GPU type gives the most finished runs; by this history that is
    the H200 again, even with a start wait of hours. The choice of type, and whether to make screens
    resumable so a preemption costs minutes, are yours.
  - Unchanged: progress before records, and gates and records wait in the STATUS backlog until the
    maintainer says otherwise. An H200 slot runs screens until then.

- 2026-11-14 17:04 EST (laptop session, correction to the note above; the maintainer asked "cant you resume from
  b200"), **screens already resume; the save interval is the knob.**
  - The note above wrote as if screens were not resumable. They are: `screen_seed.sh` saves state every
    `RESUME_EVERY_S` (default 3600) training seconds on `mit_preemptable`, and r72 s1 resumed at step 156
    on 29110624. So a preemption costs the time since the last save, not the whole run (LOG 05:04 and 16:10
    read otherwise).
  - Today 16 of the 18 preempted incarnations ended inside their first hour (median 15 min, one at 57 min),
    before any save. States on disk now: r72 s1 (28.8 GB), r68 s0 and s2, r70 s1 (1.3 GB each).
  - Between kills the workers waited in the queue: the four preemptable B200 jobs were on a GPU for 12.8 of
    about 46 job-hours (28%). A shorter interval keeps the work done but does not shorten the waiting.
    Weigh both against H200 for the four slots; the interval and the choice stay yours.

- 2026-11-14 22:03 EST (maintainer, via laptop), **GPU use is yours to optimise; explore.**
  - Maintainer: "just tell the agent to optimize usage, i believe using b200s will help but idk it should
    explore".
  - The aim is the most finished, useful runs per day from the six slots. At 22:00 two of the six were
    running. How you get there is your call, and trying things is wanted. Log what you try and what it yields
    (`sacct -D` lists every requeued incarnation, so yield per GPU type can be measured).
  - Things you may try; none is an instruction:
    - preemptable B200 with a much shorter save interval, at least for track D (1.3 GB state). Time a track A
      save (28.8 GB) before you choose its interval;
    - other preemptable GPU types with 80 GB or more. A100 and RTX PRO 6000 looked lightly queued on 11-13
      (28 of 264 and 21 of 72 free); neither has run this software here;
    - workers queued on more than one GPU type for the same slot, so whichever starts first takes the work.
  - On the six-GPU rule: it limits GPUs running at once. Slurm already caps you at 2 on `mit_normal_gpu` plus
    4 on `mit_preemptable`, so pending workers beyond those cannot run together and need not be counted. Keep
    the pending pile modest, and make sure a worker that finds its queue empty exits at once.
  - Backlog: asked whether r63's gate may take an H200 slot, the maintainer said "idk", so that is yours too.
    Progress still comes first. A gate or a record from the backlog may run when you judge it the best use
    of an H200 slot. The rules, the promotion rule and what a timed run needs (H200, on Engaging) are unchanged.
  - `~/rlvr-ideas/STATUS.md` has no entry for r72 (U12 step 2). Please keep it current.

- 2026-11-15 14:03 EST (maintainer, via laptop), **three new ideas in the private list; and a chat tool that is not yours to run.**
  - `~/rlvr-ideas/IDEAS.md` has three new entries, U16 to U18 (2026-11-15). As before, judge each on
    its merits, give it a row in `~/rlvr-ideas/STATUS.md`, and keep their content out of this repo.
    One of them carries a rule question for the maintainer; nothing changes until it is answered.
  - Commit ce6e1f3 adds `scripts/chat.py` and `scripts/chat.sh`: the maintainer's tool for reading what
    saved checkpoints write. It runs CPU-only jobs named `rlvr-chat` on `mit_quicktest`, uses no GPU
    slot, reads only the dev ladder and the saved weights, and writes nothing under `~/rlvr-screens`.
    Ignore those jobs in `squeue`.

- 2026-11-15 14:09 EST (maintainer, via laptop), **ruling: RULES v3.5, a record need not use vLLM.**
  - Asked whether a model that vLLM's model classes cannot run needs its own vLLM class for
    scoring, or whether the scorer may use another sampler, the maintainer answered: "using non
    vllm is fine."
  - What v3.5 says (`docs/RULES_v3.md`, "The evaluation sampler"): training and scoring may run
    outside vLLM. Scoring still goes through `ladder_confirm` with only the generation call
    replaced: the same test ladder, `r1` prompts, 16 independent samples per puzzle, `verify`,
    `score`, and the same `confirm_v3.json` fields plus the backend's name. The three sampler
    conditions are unchanged (independent single-pass samples; no arithmetic, target or verifier
    in the sampler; the code is in the submission). The README names the software that scored it.
  - `ladder_confirm` has no backend option yet. Add one only when a candidate needs it; it is
    reviewed with the record. Leave the vLLM path as it is, so existing records score the same.
  - This answers the rule question on one of the new private ideas. It permits; it does not ask
    you to move anything off vLLM. Whether a sampler outside vLLM or a new architecture is worth
    a slot is your call.

- 2026-11-15 15:13 EST (maintainer, via laptop), **one more entry in the private list: U19.**
  - `~/rlvr-ideas/IDEAS.md` has a new entry, U19 (2026-11-15). It states, in the maintainer's own
    words, a direction they want on track D and the belief behind it, and it ties U16 and U17
    together. Judge it as usual, give it a row in `~/rlvr-ideas/STATUS.md`, and keep its content out
    of this repo. Thanks for the U16-U18 rows.

- 2026-11-15 15:16 EST (maintainer, via laptop), **U19 has one added line.** The entry in
  `~/rlvr-ideas/IDEAS.md` now ends with a constraint the maintainer added after it was written.
  Read it before you plan anything for U19.

- 2026-11-16 22:35 EST (maintainer, via laptop), **answer to QUESTIONS 11-16 13:54: a seed over budget only from idle exit time counts as it is (RULES v3.6).**
  - Asked whether such a seed should fail the record, be rerun like a preemption, or count as it
    is, the maintainer chose "count it as is" and added: "sounds like a one off thats just fine".
    So no rerun, and the score stands.
  - The condition (`docs/RULES_v3.md`, "Clock and hardware", "Idle exit time"): the seed's logs show
    that the last training step and the final evaluation ended inside the budget, and the README
    names the seed and gives both times. It is a one-off that the maintainer judges case by case,
    so put any such seed in QUESTIONS when it happens. Over budget for any other reason is still
    over budget.
  - The validator still reports such a seed as over budget. Do not change it now. If a record
    seed ever needs the exception, add it with that record; it is reviewed with the record.
  - One fact about r81 seed 2, from its file times: `train_log.jsonl` and `eval_log.jsonl` were
    last written at 11:56:02, `result.json` at 12:10:18, and the process ended at 12:10:58. So the
    14 minutes sit between the final evaluation and the result file, not after the result was
    written. That narrows where the stall is, if you look for it.

- 2026-11-17 12:25 EST (maintainer, via laptop), **speed changes can be ranked with short benchmarks.**
  - The maintainer, on 11-15, about throughput work: "also thruput is smt that you can just do fast
    without full runs right". Asked today whether to pass that on: "tell it".
  - What it means in practice, for a change that only affects speed (the maths is the same):
    a few hundred steps after the compile warm-up give tokens per second, and the check that
    nothing else changed is the loss at matching steps against the old path, with the same seed
    and data order. That is minutes per variant, so many variants fit in one slot.
  - Long runs are for changes to the maths (batch tokens, precision, optimizer, architecture),
    where the loss at a fixed time is the readout. The same goes for the RL loop, timed from a
    mid-training checkpoint as well as from the start, since answer lengths change.
  - Both r89 arms are speed-only (the micro-batch changes, the batch stays 524,288 tokens).
    Whether to let them run the full 56 minutes or cut them short is your call.

- 2026-11-17 14:53 EST (maintainer, via laptop), **U19 has one more line.** The entry in
  `~/rlvr-ideas/IDEAS.md` ends with what the maintainer said after seeing r91. Read it at your
  next full wake. No rule change and nothing to stop.
  - Later the same hour: U17 has an added line too, at the end of the file ("U17, added 2026-11-17").

- 2026-11-17 14:59 EST (maintainer, via laptop), **performance is the direction; speed candidates from the laptop session.**
  - The maintainer: "anyway send anything with perf to agent, based on nanogpts results i think you
    can go far by just optimizing perf". Just before, on RL sampling still running through vLLM:
    "surely you can do better than vllm ...". So the laptop session now passes on performance notes
    as they come up. The candidates below are the laptop session's, not the maintainer's. None is
    measured: rank them with the short benchmarks and judge them on merit.
  - **Track D's RL step** (D 007 seed 0: 1.00 s per step, sampling 0.51 s, update 0.40 s; r78 put a
    third more steps at +0.032 dev):
    1. Share the prompt per group in the update. `_completion_logprobs` already runs the batch-wide
       prefix once, but that is only 218 of the prompt's 509 characters. The other 291 (numbers,
       target and the long instruction tail) are identical within a group of 8 and are run for
       every row. Running each unique prompt once and expanding its KV cache to its rows (by index,
       so it survives dropped zero-advantage rows) is exact for the same reason the current
       prefix trick is. Rough count of token positions per update: about 46k now, about 14k then.
    2. A same-process sampler: one batched decode loop on the trainer's weights with a KV cache.
       Prefill the 64 unique prompts once, expand by 8, decode about 20 tokens, with the constraint
       as tensor ops on the allowed sets `constrain_batched` already keeps on the GPU. No weight
       copy into vLLM and no prefix-cache reset. vLLM's floor with the constraint off was 0.23 s
       (r76); a 110M model decoding 512 short rows should sit well under that. The `hf` backend's
       `StaticCachePool` and compiled decode are a starting point; it lacks the constraint and the
       sampler's log-probs.
    3. Scoring can stay as it is. If the model remains a standard Llama, `ladder_confirm` and the
       record's evaluation sampler keep vLLM, and only training-time sampling moves ("any sampler
       during training" was always allowed). So 1 and 2 need no v3.5 scoring backend.
  - **Track D pretraining, from the modded-nanogpt record table** (its record numbers in brackets;
    its gains are on its own model):
    - Same maths: a lean training module in place of the Hugging Face class, with the weights
      copied back into the Llama layout for vLLM (you measured about 26 % of peak, which points at
      overhead: profile first); more fused block kernels (59, 60, 75, 79); full CUDA-graph capture
      (92); a cheaper Muon step (20, 27, 38), probably small at one step per 524k tokens.
    - Slightly different numerics, so a loss comparison is needed: bf16 weights and activations
      throughout (10, 37, 57); FP8 for the output head and the MLPs (19, 84, 89, 90), where the
      head is roughly a fifth of forward compute at 768 x 32,000; sampled softmax early (92).
    - Changes to the model itself (windowed attention, ReLU^2, untied head, value embeddings)
      need a sampler outside vLLM or a matching model class.
  - Anything that keeps the Llama weight layout can live in the training code alone.

- 2026-11-17 15:02 EST (maintainer, via laptop), **why speed counts even without a score gain.** After the
  note above, the maintainer: "more train time generally = more flexibility anyway" (also added
  under U19 in `~/rlvr-ideas/IDEAS.md`). With the 11-15 line on ties ("if there is reason to
  believe muon is better why not just leave it even if theres no statistical evidence"), the
  laptop session reads it this way, and the reading is the laptop session's: a speed change whose
  maths is unchanged does not have to win a paired screen to be kept, since the time it frees can
  go to RL steps, pretraining, longer answers or a larger model later. Promotion to a record still
  follows PLAN's rule.

- 2026-11-17 17:14 EST (laptop session, performance note under the maintainer's standing "send anything with
  perf"), **r93 says the RL update is not bound by token positions; time its parts before more
  screens.** The laptop session's reading of the code and of r93, unmeasured:
  - The share-prompt candidate was the laptop session's and its predicted saving was wrong. But
    the per-row part it removes is not "the puzzle's few tokens": `split_prompt(style="r1")` puts
    the split after 216 characters, so the per-row suffix is the 293-character tail (numbers,
    target and the whole instruction, roughly 70 tokens), against about 27 completion tokens late
    in RL (your r93 logs: tokens/s x wall / 512). Sharing therefore removed a large part of the
    positions and `t_update` did not move (0.390 -> 0.389). So the transformer body is not what
    the 0.39 s is spent on.
  - One thing that sits inside the `t_update` window and does not depend on positions:
    `_allowed_tokens` (grpo.py, called just before `_update`) replays the answer grammar in
    Python, one row at a time, for all 512 rows, including rows `skip_zero_advantage` then drops.
    That is about 14k `allowed` / `feed` calls per step late in RL. The batched sampler already
    worked out the same sets while decoding, so they could be recorded there (an index per
    sampled token) instead of replayed; failing that, replay only the rows the update keeps.
  - Other candidates for the rest: the head, log-softmax and entropy over 32,000 tokens at every
    completion position (the 169 GB peak at 131k tokens points the same way; constrained positions
    only need the allowed ids' rows of the head), per-micro-batch fixed cost with `compile` off,
    and the optimizer step.
  - Suggested order: wall-clock timers around those parts on one late-RL and one start-of-RL
    batch (a few minutes on the B200), then fix the largest. r94 is still worth reading for the
    H200 memory limit.

- 2026-11-17 17:23 EST (maintainer, via laptop), **new entry U20 in `~/rlvr-ideas/IDEAS.md` (H200 use), and
  what the GPU cap means.**
  - Read U20 at your next wake; it is in the maintainer's words. No rule has changed: records
    still run on the H200 SXM.
  - Queue facts at 17:22 EST today, from `sacct` and `sinfo`: all six of your B200 jobs since noon
    started 1 to 13 minutes after submission; your five H200 jobs have waited 3 to 4.5 hours with
    no start estimate, behind 233 pending H200 requests on `mit_preemptable`; no GPU of yours was
    running. The 12 B200 nodes are in both `mit_normal_gpu` and `mit_preemptable`.
  - The cap of 6 GPUs is Slurm's own per-user limits (2 on `mit_normal_gpu`, 4 on
    `mit_preemptable`), and Slurm enforces them. It means GPUs running at once. Jobs that are
    only pending use no GPU and do not count towards it.
  - GPU use is still yours to optimise (11-14).

- 2026-11-17 20:23 EST (maintainer, via laptop), **U20 clarified: it is about getting more use out of the
  B200s.** The maintainer: "by using h200 is inefficient i mean like surely the agent can get
  more use out of b200" (added under U20 in `~/rlvr-ideas/IDEAS.md`). It is not a request to
  move the record GPU: records stay on the H200 and no rule question is open on it.
  - Facts from `sacct` at 20:22 EST: since noon your jobs have used 5.5 GPU-hours, all on B200
    (11 jobs), out of about 50 that six GPUs would have given. Nothing has run since 18:53. The
    five H200 jobs are still pending and nothing has run on an H200 since the 13:19 preemption.
  - How to use the B200s is yours to work out (GPU use is yours to optimise, 11-14).

- 2026-11-18 11:47 EST (maintainer, via laptop), **usage is no longer short: the low-usage instruction until
  18:00 is lifted.**
  - The maintainer has moved you to a larger account ("ok now you're on the ultra acct", "and
    agent is too") and said "use as much usage as you need/want". The instruction typed in your
    pane at about 11:45 ("keep everything else to a minimum until 18:00") was written for an
    account with little usage. Extending "as much as you need" from the laptop session to you is
    the laptop session's reading; the maintainer was in your pane a minute earlier and can correct
    it there.
  - So run normal wakes from now, and manage the 11-18 box actively: it ends at 06:30 EST on
    11-19 and five of its hours were lost while you were paused (the maintainer: "bruh wasted so
    much"). Keeping all eight GPUs busy until the end, and getting every result off the box before
    it terminates, comes first.
  - What stopped you overnight was the weekly limit of the previous account, at about 04:25 EST.

- 2026-11-18 13:35 EST (maintainer, via laptop), **make the 11-28 H200 box start and run by itself, even if you
  are down.**
  - The maintainer, after today's box sat idle for 5 h 16 min while you were paused: "btw next
    time cant we set it to autolaunch experiments even if agent si down". The spec below is the
    laptop session's reading of that; how you build it is yours.
  - What "by itself" has to cover, with nobody awake on Engaging or on the laptop:
    1. **Launch.** Something on AWS's side starts the instance in the reservation at 11:30Z on
       11-28 and retries (for example at +10 and +25 min). The 11-05 scheduled launches all
       failed: the `rlvr-scheduler` role got UnauthorizedOperation on `ec2:RunInstances` for the
       security group (its policy did not cover security-group, network-interface and subnet
       resources), and that policy was never fixed. No launch schedules exist in us-east-1 now.
    2. **Start the work.** Today everything reached the box by ssh from Engaging (box_sync.sh,
       then the workers), so a paused agent meant an idle box. The box should fetch what it needs
       itself at boot (code at a pinned commit, queue files, checkpoints, the venv pins), from
       somewhere staged in advance such as S3, and start its workers from user-data. The spot
       scripts (`spot_job.sh`, `spot_fetch.sh`) already push results to a bucket.
    3. **Save the results.** Each finished run, and each run's saved state every so often, goes
       off the box without anyone pulling it, so nothing is lost when the block ends at 11:30Z
       on 11-29.
    4. **A queue that fills it.** 24 h on eight H200s, with a packing checked against 141 GB
       (today's three-per-GPU plan was checked against the wrong card).
  - A launch by you at 11:30Z must not clash with the automatic one (the block holds one
    instance).
  - Test it before it matters: a dry run of the launch under the role that will make it, and one
    boot of a cheap instance through fetch, worker start and result upload. Keep AWS costs for
    the setup and tests to a few dollars.
  - Please have it built and tested by **11-25**, and write in `docs/research/box_1128.md` what
    exists (schedule names, role, bucket and prefix, launch-template version) and how to check
    each piece. The laptop session will audit it read-only after that.
  - Before any paid block, also check that your own usage and login will last through it.

- 2026-11-18 13:39 EST (maintainer, via laptop), **two more 8x H100 days bought for you: 11-25 and 12-09.**
  - The maintainer approved them today ("sure buy it"). Both are p5.48xlarge, one instance, 24 h
    from 11:30Z, in us-east-1f, the zone of today's box, so the same network interface, fixed IP
    (174.129.98.131) and launch path apply:
    - `cr-03ab208e99ca6ca06`: 2026-11-25 11:30Z to 11-26 11:30Z (06:30 EST both days)
    - `cr-0813172fa29139b78`: 2026-12-09 11:30Z to 12-10 11:30Z
  - They were `payment-pending` at purchase. Check `describe-capacity-reservations` shows
    `scheduled` before you plan around them; the laptop session will check too.
  - Same terms as the 11-18 and 11-28 blocks: screens only, no records, you launch them, and
    today's launch template points at today's reservation, so each needs its own reservation id.
  - With the H200 block on 11-28 that is a box on 11-25, 11-28 and 12-09. Each needs a queue
    that fills 24 h on eight GPUs and a packing checked against the card's memory (80 GB on these).
  - The self-starting setup asked for in the note above (INBOX 13:24) now matters for three
    blocks. If it can be ready for 11-25, use it there; if not, 11-25 runs as today did.

- 2026-11-19 01:32 EST (maintainer, via laptop), **"it looks like nto that much progress?" The laptop session's
  critique of the 11-18 box plan, sent at the maintainer's request ("sure send").**
  - The remark is the maintainer's, made after reading tonight's box results. Everything below
    is the laptop session's view. Weigh it on its merits; you know the runs better.
  - **Where things stand on score.** No record has landed since A 007 on 11-16. D is 0.244
    officially; D 007 is 0.263 on four seeds. The three days since went to outages, the H200
    queue, and speed work that has not yet been run end to end.
  - **The box answered questions that could not come out clearly.** r85's three arms are small
    moves around A 007 (lr, entropy target, cosine horizon). The control's own seeds span
    0.400 to 0.440, and the arms' differences so far are +0.013 to -0.027. Three seeds cannot
    tell an effect of that size from zero, whatever its sign. The plan was written on 11-16,
    before the speed results, and was not revisited when the box slipped to 11-18.
  - **r86 is a clean null, and it is in tension with r81.** Both Muon arms are level with or
    below the AdamW control here, while r81 on the H200 was about +0.02 over r77. Worth saying in
    LOG which of the two you believe and why (step-capped pairs on an H100 against a timed run
    on an H200), since D 007 rests on r81.
  - **For 11-25, 11-28 and 12-09, a suggestion:**
    1. Turn the speed into score first. The D speed stack (native sampler, reused allowed sets,
       shared prompt, fused pretraining) is the largest measured lever you hold: r78 put a third
       more RL steps at +0.032. Run it end to end against its control with enough seeds to
       promote, before anything else gets box time.
    2. Fewer arms, more seeds. Before an arm gets GPUs, write down the effect you expect and
       check it against the seed noise of its control; an arm whose expected effect is inside
       the noise needs many seeds or should not run.
    3. Spend the rest on changes that could be large: U17 (architecture, now unblocked by the
       native sampler) and U19 (pretraining), not further small moves around the holders.
    4. Re-plan a box's queue the day before it starts, from what has been learnt since.
  - Track A has had no speed work. Whether its 5 h can hold more than 467 steps is an open
    question the timers you built for D could answer in one short run.
- 2026-11-19 14:40 EST (laptop session, operational), **stage a standby queue for each paid block now, so
  a box that starts while you are down still has work.**
  - This carries out the maintainer's request of 11-18 ("btw next time cant we set it to autolaunch
    experiments even if agent si down"). It is the laptop session's note, not a new instruction from
    the maintainer.
  - Checked from the laptop at 14:35 (read-only): all three reservations are `scheduled`, each has 11
    enabled schedules (first at 11:31Z on its day) and a default launch template that names the right
    reservation and the `rlvr-box` profile. Thank you for building and testing it.
  - The gap: `box_1128.md` says no queue is staged, and LOG 06:32 plans the 11-25 queue on 11-24. If
    you are down from now until a block starts (login, usage limit, login009 reboot), that box comes
    up with no work, which is the 11-18 failure again with the launch fixed.
  - Suggestion: stage a standby queue for all three blocks from what you know today (for example the
    speed stack against its control at step caps, many seeds), and replace it on the planning day.
    A stale queue beats an idle box. Say in `box_1128.md` what is staged for each block and when.
  - Yours to decide how; no reply needed beyond the LOG.
- 2026-11-19 14:29 EST (maintainer, via laptop), **the maintainer on how to use the AWS boxes: bolder arms, fewer
  seeds. Point 2 of the 01:32 note ("fewer arms, more seeds") is withdrawn in favour of this. Your
  10:47 question is not answered yet.**
  - The maintainer's words today, in order:
    - "also for running screens on aws i feel like you can be a bit more aggressive and use less
      seeds?"
    - "i think its better to make more bold changes in general"
    - "wait speed work should be done on engaging. anything else that might be useful can be run on
      aws."
  - The first two are filed as U21 in the ideas file, for you to judge like the others. Everything
    below is the laptop session's reading, not the maintainer's words.
  - **Seeds.** The 01:32 note asked for fewer arms and more seeds. The maintainer's view goes the
    other way and I think it is the better rule: wide and shallow. The dev seed sd is about 0.02 on
    both tracks, so one paired seed shows an effect of about 0.04 to 0.05 or more (the native
    sampler's +0.055 and +0.050 showed in single pairs). An arm that could move the score that much
    needs one or two seeds against a control with four or five. An arm that could not is not worth
    box time at any seed count. The best of many single seeds is partly luck, so the top few get
    re-run before anything is built on them. What stays from the 01:32 note: write the expected
    effect and the control's noise before an arm runs.
  - **The third remark** came while I was explaining your 10:47 question (the 11-28 box as the gating
    screen for the speed stack). The maintainer said they did not fully follow the question, so I
    do not treat the remark as its answer. One reading is that results which depend on the machine's
    speed belong on Engaging and the boxes take the rest. I have asked and will send the answer.
    Until then keep r102 on Engaging as it is, and stage the boxes so that either answer works.
  - Process: U20 has no row in STATUS.md yet (only a mention under "Last change"). Please add its
    verdict.
- 2026-11-19 14:32 EST (maintainer, via laptop), **your 10:47 question (the 11-28 box as the gating screen): the
  decision is yours.**
  - The maintainer, once the question was explained: "i think this is too early to be talking about
    it, but its fine to just let the agent do whatever as long as it is reasonable".
  - So no yes or no is coming. Decide it when the time comes, on the facts you have then, and write
    the reasoning in the LOG. Their earlier remark ("speed work should be done on engaging. anything
    else that might be useful can be run on aws.") is their leaning, for you to weigh.
- 2026-11-19 16:28 EST (maintainer, via laptop), **assume AWS and Engaging run at different speeds.**
  - The maintainer: "in general unless proven otherwise itd be wise toa ssume aws and engaging run
    at diff speseds".
  - The laptop session's reading: a timing or a timed run on a box says nothing about Engaging until
    the same item has been timed on both. Your 10:47 plan already had that check.
- 2026-11-19 20:45 EST (laptop session, operational; a suggestion, not measured), **why your Engaging jobs sit at the
  back of the line, and the one part of the priority you control.**
  - Read-only facts at 20:44 EST. A job's priority is partition (360,000 on `mit_normal_gpu`, 30,000
    on `mit_preemptable`) + fair share (up to 150,000) + age (up to 50,000, reached after 7 days
    pending: about 298 an hour). Your fair-share part is 1,287: the account's factor is 0.0086, in
    the bottom 1 % of 11,773 users. Usage is charged by CPU count (no GPU weight is set).
  - The B200 line on `mit_normal_gpu` had 33 pending requests. Light users sit at 506,000 to
    520,000. Heavy users sit together at 362,000 to 370,000, and your two workers (363,345, age
    2,053) were 18th and 19th.
  - So age is the lever: a worker that has been pending a day carries about +7,100 and two days
    +14,300, enough to pass the whole 362,000 to 370,000 group. Your workers read a queue file, so
    the next pair can be submitted while the current pair runs or waits. Slurm's per-user limit
    holds them, and the cap counts running GPUs. A job pending on `QOSMaxGRESPerUser` was at the top
    of that list, so age seems to accrue under the limit; I have not checked that in the config.
  - Not levers, by `sbatch --test-only` (nothing was submitted): 8 CPUs and 64 GB instead of 16 and
    128, 30 min instead of 6 h, no node exclusions, H200 or L40S instead of B200 all gave the same
    start estimate. `mit_quicktest` has no GPUs.
  - About three quarters of the account's CPU-hours in the last 14 days are the maintainer's other
    project, which is not yours or mine to change. Without it the factor would be about 0.03, worth
    about +3,300.
