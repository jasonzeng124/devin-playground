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

- 2026-10-22 (laptop), URGENT: node1800 is a black-hole node. Worker 26424658 ran there on 10-21 and every GPU item it touched FAILED: the 5 D2 items (d00_dm26/dm51) after ~113 min (pretraining finished, then the RL start died), and 28 other items within ~20-45 s. The traceback (e.g. calibration/research_r2/d00_fp32_lr3e-6_skip/seeds/4/stdout.log) is triton failing to compile cuda_utils.c with gcc, probably a broken or full /tmp on that node. It drained main, long and r1 to empty. To do:
  (1) Exclude node1800 from all workers. New submissions get --exclude=node1708,node1709,node4508,node1800. For pending ones, ask before any scontrol update, since auto mode blocked one; if it's blocked, note it in QUESTIONS.
  (2) Requeue the 33 failed items listed by worker 26424658 in queue/worker.log. First move their partial seed dirs to extra/ per RULES/run_seed_v3 conventions.
  (3) Make queue_worker stop (and leave the item queued) after 2 consecutive failures shorter than 5 min, so one bad node can't drain a queue again.
