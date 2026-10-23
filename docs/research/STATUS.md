# Status

(maintained by the research agent; see AGENT.md)

**Promotion: none yet.** (When an arm meets the promotion rule, this line names the arm, its
config path and its dev numbers, and the record folder's config.json is committed; the record
then runs on AWS via the laptop session or on Engaging, whichever starts sooner.)

Updated 2026-10-22 23:55 EDT.

## Running

- Engaging: nothing runs. The queues drained on 10-21/22 (35 items failed on Ubuntu 24 nodes; LOG
  2026-10-22 night). The last GPU item ended at 10:27 on 10-22.
- Workers: 6 on mit_preemptable, all pending (26644470 `long:main`, 26644471-75 `main:long`;
  `RUNNER=scripts/research/run_item.sh`, RESERVE_MIN 330, H100 80GB HBM3; node1800 and the NVL nodes
  excluded). Slurm's start estimate is ~00:20 on 10-24 (2,374 jobs pending). None on mit_normal_gpu:
  its estimate is 10-25 09:00 for H100 and H200 alike.
- Workers now check gcc before popping. On Ubuntu nodes they add the multiarch paths; if gcc
  still fails they exit with nothing popped. After two failures in a row under 5 min, a worker
  puts the item back and exits.
- AWS (laptop session): a01_skip seeds 0-2 on on-demand p5.4xlarge (run a01_skip-aws-od-20261022),
  dev only; results not here yet.

## Queues (~105 GPU-hours)

- `main` (17): a01_fp32_lr1e-6_skip_cos seeds 1-2; r4 dm26_fp32_lr3e-6_skip seeds 0-2;
  d00_fp32_lr3e-6_skip seeds 0-2; r4 dm51_fp32_lr3e-6_skip seeds 0-2; a01_skip seeds 3-4;
  then seeds 3-4 of a01_fp32_lr1e-6_skip_cos and d00_fp32_lr3e-6_skip.
- `long` (19): d00_fp32_lr3e-6 seed 1 (resumes at 180 of 234 min), samp_a001_t025 seeds 0 and 2,
  c90 collapse screens seeds 0-1, then seeds 2-3.
- r4 arms run the RL stage from the saved D2 pretrained models (`screen_pretrained`).

## Best so far

- Track A: `a01_fp32_lr1e-6_skip_cos` seed 0 at dev **0.300** (A 001's seed 0: 0.203). After ~100 min
  of one-guess answers it regrew long searches. The 3-seed form needs seeds 1-2 >= 0.270.
- Track D: fp32 lr 3e-6 RL on D 000's pretraining, 0.138 mean over 5 seeds (D 000: 0.119; +0.019, no
  promotion; seed 1 still resumes). D2 dm26 seed 0 with D 000's RL: 0.118 (D 000 seed 0: 0.087).
- Holders: track A 001 (test 0.2481, dev 0.240), track D 000 (test 0.1166, dev 0.119). Promotion
  needs dev mean >= 0.270 (A) / 0.149 (D), with 4 of 5 seeds above the holder's dev mean and at most
  1 collapsed, or 3 seeds all that high and none collapsed. "Collapsed" = diverged, or one-guess
  below the holder's dev mean (clarified 10-22, PLAN.md; QUESTIONS.md asks).
- Sampler: T 0.5 (T 0.25 and think budget 960 add nothing).

## Next

- About 6 h after the workers start: skip_cos seeds 1-2. If both are >= 0.270 and not collapsed,
  A promotes: commit `records/v3/track_a/002_<slug>/config.json` (A 001 + fp32 lr 1e-6 + skip +
  cosine, sampler T 0.5) and flag it above. Otherwise seeds 3-4 move up for the 5-seed form.
- About 5-10 h after the start: dm26 and d00 skip seeds 0-2. A D arm whose 3 seeds are all >= 0.149
  promotes. The record pipeline would be D 000's (or DM-math) pretraining + fp32 lr 3e-6 + skip.
- a01_skip AWS seeds 0-2 whenever the laptop commits them (3-seed check; seeds 3-4 leave `main` if
  they settle it).
- Successor workers about 2 days after the current ones start.

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q`: without it the two-rank tests in
  `tests/test_distributed.py` fail on the login node ("libgomp: Thread creation failed").
- Eval arms: a dir with `eval.json` queued like a screen runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`, `dev_ladder_eval.py`); `summarize_screens.py` prints them.
- Track D pretraining speed varies by node: 224k tokens/s on node4109 (4 of our jobs at once)
  against 372k on node3009 and 359-373k in D 000's record. Pretraining is time-budgeted, so D2
  results must be read with their tokens/s. The 6 saved D2 models all ran at ~370k.
- Ubuntu 24 nodes (8 of ~35 H100 nodes on mit_preemptable): `module load` fails there (lmod path),
  and gcc needs the multiarch paths, which the worker now adds.
