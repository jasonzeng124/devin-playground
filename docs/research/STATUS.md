# Status

(maintained by the research agent; see AGENT.md)

**Promotion: none yet.** (When an arm meets the promotion rule, this line names the arm, its
config path and its dev numbers, and the record folder's config.json is committed; the record
then runs on AWS via the laptop session or on Engaging, whichever starts sooner.)

Updated 2026-10-23 03:50 EDT.

## Running

- Workers: 6 on mit_preemptable (`RUNNER=scripts/research/run_item.sh`, RESERVE_MIN 330, H100 80GB
  HBM3; node1800 and the NVL nodes excluded). 4 run; 26644474-75 pend on `QOSMaxGRESPerUser`: the
  mit_preemptable QOS allows 4 GPUs per user (`sacctmgr show qos mit_preemptable`: gres/gpu=4), so
  they only start when a running worker ends. Usable GPUs are 4 there + 2 on mit_normal_gpu, whose
  next H100 start is estimated at 10-25 17:00 (H200 10-25 13:50).
  - 26644470 (`long:main`, node2701, since 23:59): finished d00_fp32_lr3e-6 seed 1 (0.143), the two
    T 0.25 evals and c90_mal0 seed 0 (0.322 at 90 min); now c90_mal0 seed 1, done ~04:10.
  - 26644471 (`main:long`, node2701, since 23:59): a01_fp32_lr1e-6_skip_cos seed 1, done ~05:00.
  - 26644472 (`main:long`, node2700, since 00:26): r4 dm26_fp32_lr3e-6_skip seed 0, done ~04:30.
  - 26644473 (`main:long`, node2700, since 00:26): a01_fp32_lr1e-6_skip_cos seed 2, done ~05:30.
- Both nodes run Ubuntu 24. The workers' gcc check added the multiarch paths there, and all four
  items got past vLLM's startup compile, which is where the 10-21 failures died. The fix works.
- Successors 26649456-59, each `afterany` on one running worker (26644470-73), same queues, so
  they start as the current workers end (~10-24 evening) and the cap holds.
- AWS (laptop session): a01_skip seeds 0-2 on on-demand p5.4xlarge (run a01_skip-aws-od-20261022),
  dev only; results not here yet.

## Queues (~95 GPU-hours)

- `main` (17): r5 a01_fp32_lr1e-6_skip_cos_mal0 seeds 0-2 (290 min, weights kept);
  r4 dm26_fp32_lr3e-6_skip seeds 1-2; d00_fp32_lr3e-6_skip seeds 0-2;
  r4 dm51_fp32_lr3e-6_skip seeds 0-2; a01_skip seeds 3-4; then seeds 3-4 of
  a01_fp32_lr1e-6_skip_cos and d00_fp32_lr3e-6_skip.
- `long` (14): c90 collapse screens (t12, cc1e-3, pos01 seed 0, then seed 1s, then seeds 2-3).
- r4 arms run the RL stage from the saved D2 pretrained models (`screen_pretrained`).

## Best so far

- Track A: `a01_fp32_lr1e-6_skip_cos` seed 0 at dev **0.300** (A 001's seed 0: 0.203); at 02:40 seed 2
  was at 0.383 (131 min) and seed 1 at 0.222 (154 min, still one-guess). The 3-seed form needs
  seeds 1-2 >= 0.270. `c90_mal0` (the same with malformed reward 0) reached **0.322 in 90 min** on
  seed 0, with 62 % of eval answers truncated; r5 runs it for 290 min.
- Track D: fp32 lr 3e-6 RL on D 000's pretraining, 0.137 ± 0.009 over 5 seeds (D 000: 0.119; +0.018,
  no promotion). D2 dm26 seed 0 with D 000's RL: 0.118 (D 000 seed 0: 0.087).
- Holders: track A 001 (test 0.2481, dev 0.240), track D 000 (test 0.1166, dev 0.119). Promotion
  needs dev mean >= 0.270 (A) / 0.149 (D), with 4 of 5 seeds above the holder's dev mean and at most
  1 collapsed, or 3 seeds all that high and none collapsed. "Collapsed" = diverged, or one-guess
  below the holder's dev mean (clarified 10-22, PLAN.md; QUESTIONS.md asks).
- Sampler: T 0.5 (T 0.25 and think budget 960 add nothing).

## Next

- ~05:00-05:30: skip_cos seeds 1-2. If both are >= 0.270 and not collapsed, A promotes: commit
  `records/v3/track_a/002_<slug>/config.json` (A 001 + fp32 lr 1e-6 + skip + cosine, sampler T 0.5)
  and flag it above. Otherwise seeds 3-4 move up for the 5-seed form.
- From ~04:30 into 10-24: dm26 and d00 skip seeds 0-2. A D arm whose 3 seeds are all >= 0.149
  promotes. The record pipeline would be D 000's (or DM-math) pretraining + fp32 lr 3e-6 + skip.
- ~04:10: c90_mal0 seed 1 (does 0.322 at 90 min replicate?).
- ~10:00-10:30: r5 mal0 seeds 0-2 (290 min). Then dev sampler evals on their weights: T 0.5 with and
  without a think budget.
- a01_skip AWS seeds 0-2 whenever the laptop commits them (3-seed check; seeds 3-4 leave `main` if
  they settle it).
- Successors for the 2 pending workers once they start.

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q`: without it the two-rank tests in
  `tests/test_distributed.py` fail on the login node ("libgomp: Thread creation failed").
- Eval arms: a dir with `eval.json` queued like a screen runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`, `dev_ladder_eval.py`); `summarize_screens.py` prints them.
- Track D pretraining speed varies by node: 224k tokens/s on node4109 (4 of our jobs at once)
  against 372k on node3009 and 359-373k in D 000's record. Pretraining is time-budgeted, so D2
  results must be read with their tokens/s. The 6 saved D2 models all ran at ~370k.
- Ubuntu 24 nodes (8 of ~35 H100 nodes on mit_preemptable): gcc needs the multiarch paths, which
  the worker now adds. `module load` fails there because lmod isn't installed (probe 26644751 on
  node2801), so `cuda/13.0.1` isn't loaded; vLLM and training start without it.
