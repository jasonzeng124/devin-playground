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
