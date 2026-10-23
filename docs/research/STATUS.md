# Status

(maintained by the research agent; see AGENT.md)

**Promotion: none yet.** (When an arm meets the promotion rule, this line names the arm, its
config path and its dev numbers, and the record folder's config.json is committed; the record
then runs on AWS via the laptop session or on Engaging, whichever starts sooner.)

Updated 2026-10-23 07:05 EDT.

## Running

- Workers: 6 on mit_preemptable (`RUNNER=scripts/research/run_item.sh`, RESERVE_MIN 330, H100 80GB
  HBM3; node1800 and the NVL nodes excluded). 4 run; 26644474-75 pend on `QOSMaxGRESPerUser`: the
  mit_preemptable QOS allows 4 GPUs per user (`sacctmgr show qos mit_preemptable`: gres/gpu=4), so
  they only start when a running worker ends. Usable GPUs are 4 there + 2 on mit_normal_gpu, whose
  next H100 start is estimated at 10-25 17:00 (H200 10-25 13:50).
  - 26644470 (`long:main`, node2701): r4 dm26 skip seed 1 (~09:50; 0.090 at 53 min, seed 0 had 0.118).
  - 26644471 (`main:long`, node2701): r5 a01_fp32_lr1e-6_skip_cos_mal0 seed 1, done ~10:05 (0.272 at 75 min).
  - 26644472 (`main:long`, node2700): r5 mal0 seed 0, done ~09:20 (0.380 at 117 min, 59 % truncated).
  - 26644473 (`main:long`, node2700): r5 mal0 seed 2, done ~10:30 (0.252 at 58 min).
- Both nodes run Ubuntu 24. The workers' gcc check added the multiarch paths there, and all four
  items got past vLLM's startup compile, which is where the 10-21 failures died. The fix works.
- Successors 26649456-59, each `afterany` on one running worker (26644470-73), same queues, so
  they start as the current workers end (~10-24 evening) and the cap holds.
- AWS (laptop session): a01_skip seeds 0-2 on on-demand p5.4xlarge (run a01_skip-aws-od-20261022),
  dev only; results not here yet.

## Queues (~70 GPU-hours)

- `main` (14): r4 dm26_fp32_lr3e-6_skip seed 2; d00_fp32_lr3e-6_skip seeds 0-2;
  r4 dm51_fp32_lr3e-6_skip seeds 0-2; a01_skip seeds 3-4; then seeds 3-4 of
  a01_fp32_lr1e-6_skip_cos and d00_fp32_lr3e-6_skip.
- `long` (3): r4 dm26 skip seed 1, c90_cc1e-3 seed 0, c90_pos01 seed 0.
- `later` (11, read by no worker): the other c90 seeds, parked 10-23 04:45.
- r5 sampler evals (`samp_mal0_t050`, `_tb960`, `_tb768`) get queued once r5's weights exist.
- r4 arms run the RL stage from the saved D2 pretrained models (`screen_pretrained`).

## Best so far

- Track A: `a01_fp32_lr1e-6_skip_cos` seeds 0-2 at **0.300 / 0.270 / 0.472** (mean 0.347; A 001 on
  the same seeds: 0.203 / 0.237 / 0.193). The 3-seed form misses by one puzzle (seed 1: 162 of 600;
  the line is 0.2403 + 0.03 = 0.2703). `c90_mal0` (malformed reward 0, 90 min): 0.322 and 0.383; r5
  runs it for 290 min.
- Track D: `dm26_fp32_lr3e-6_skip` seed 0 at **0.152** (line 0.149; seeds 1-2 decide). fp32 lr 3e-6
  RL on D 000's pretraining: 0.137 ± 0.009 over 5 seeds (D 000: 0.119; no promotion).
- Holders: track A 001 (test 0.2481, dev 0.2403), track D 000 (test 0.1166, dev 0.1190). Promotion
  needs dev mean >= 0.2703 (A; 163 of 600) / 0.149 (D; 90 of 600), with 4 of 5 seeds above the
  holder's dev mean and at most 1 collapsed, or 3 seeds all that high and none collapsed. "Collapsed" = diverged, or one-guess
  below the holder's dev mean (clarified 10-22, PLAN.md; QUESTIONS.md asks). On answer-constrained arms "diverged" only
  means truncated searches, so it counts like one-guess (clarified 10-23 07:00, before r5 ended; QUESTIONS.md asks).
- Sampler: T 0.5 (T 0.25 and think budget 960 add nothing).

## Next

- ~09:20-10:30: r5 mal0 seeds 0-2. If they meet the promotion rule (3 seeds all >= 0.2703 and none
  collapsed), r5 is the A record candidate: commit `records/v3/track_a/002_<slug>/config.json` with
  the sampler chosen on r5's weights (T 0.5, think budget or not) and flag it above. If r5 misses,
  skip_cos seeds 3-4 go to the front for the 5-seed form (draft record config in the scratchpad).
  A "diverged" label caused only by truncated searches goes to QUESTIONS.md before any decision.
- ~09:50 and ~13:30: dm26 skip seeds 1-2. Both >= 0.149 promotes track D (3-seed form).
- a01_skip AWS seeds 0-2 whenever the laptop commits them.
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
