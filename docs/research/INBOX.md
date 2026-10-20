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
