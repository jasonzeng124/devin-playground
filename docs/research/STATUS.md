# Status

(maintained by the research agent; see AGENT.md)

**Promotion: none yet.** (When an arm meets the promotion rule, this line names the arm, its
config path and its dev numbers, and the record folder's config.json is committed; the record
then runs on AWS via the laptop session or on Engaging, whichever starts sooner.)

Updated 2026-10-23 10:00 EDT.

## Running

- Workers: 6 on mit_preemptable (`RUNNER=scripts/research/run_item.sh`, RESERVE_MIN 330, H100 80GB
  HBM3; node1800 and the NVL nodes excluded). 4 run; 26644474-75 pend: the mit_preemptable QOS
  allows 4 GPUs per user (`sacctmgr show qos mit_preemptable`: gres/gpu=4), so they only start when
  a running worker ends. Usable GPUs are 4 there + 2 on mit_normal_gpu, whose next H100 start is
  estimated at 10-25 17:00 (H200 10-25 13:50).
  - 26644470 (`long:main`, node2701): r6 `a01_fp32_lr1e-6_skip_cos_tb768` seed 0 (09:35 -> ~14:35).
  - 26644472 (`main:long`, node2700): `a01_fp32_lr1e-6_skip_cos` seed 3 (09:18 -> ~14:20).
  - 26644471 (`main:long`, node2701): r5 seed 1 (collapsed) until ~10:05, then dm26 skip seed 2
    (-> ~14:05).
  - 26644473 (`main:long`, node2700): r5 seed 2 (collapsed) until ~10:30, then skip_cos seed 4
    (-> ~15:30).
- Ubuntu 24 nodes (node2700, node2701) work since the gcc fix.
- Successors 26649456-59, each `afterany` on one running worker (26644470-73), same queues, so
  they start as the current workers end (~10-24 evening) and the cap holds.
- AWS (laptop session): a01_skip seeds 0-2 on on-demand p5.4xlarge (run a01_skip-aws-od-20261022),
  dev only; results not here yet.

## Queues (~70 GPU-hours)

- `main` (14): dm26 skip seed 2, skip_cos seed 4, r6 seeds 1-2, r4 dm51_fp32_lr3e-6_skip seeds 0-2,
  d00_fp32_lr3e-6_skip seeds 0-2, a01_skip seeds 3-4, d00_fp32_lr3e-6_skip seeds 3-4.
- `long` (2): c90_cc1e-3 seed 0, c90_pos01 seed 0.
- `later` (11, read by no worker): the other c90 seeds, parked 10-23 04:45.
- r4 arms run the RL stage from the saved D2 pretrained models (`screen_pretrained`).

## Best so far

- Track A: `a01_fp32_lr1e-6_skip_cos` seeds 0-2 at **0.300 / 0.270 / 0.472** (mean 0.347; A 001 on
  the same seeds: 0.203 / 0.237 / 0.193). The 3-seed form misses by one puzzle (seed 1: 162 of 600;
  the line is 0.2403 + 0.03 = 0.2703). Malformed reward 0 (r5) peaks near 0.39-0.41 at 140-170 min,
  then all 3 seeds collapse (finals 0.06-0.11), so it is out.
- Track D: `dm26_fp32_lr3e-6_skip` seeds 0-1 at **0.152 / 0.143** (mean 0.148; the line is 0.149, so
  the 3-seed form is out and seeds 2-4 must average >= 0.150 for the 5-seed form). fp32 lr 3e-6 RL on
  D 000's pretraining: 0.137 ± 0.009 over 5 seeds (no promotion).
- Holders: track A 001 (test 0.2481, dev 0.2403), track D 000 (test 0.1166, dev 0.1190). Promotion
  needs dev mean >= 0.2703 (A; 163 of 600) / 0.149 (D; 90 of 600), with 4 of 5 seeds above the
  holder's dev mean and at most 1 collapsed, or 3 seeds all that high and none collapsed.
  "Collapsed" = diverged, or one-guess below the holder's dev mean (clarified 10-22, PLAN.md;
  QUESTIONS.md asks). On answer-constrained arms "diverged" only means truncated searches, so it
  counts like one-guess (clarified 10-23 07:00, before r5 ended; QUESTIONS.md asks).
- Sampler: T 0.5 (T 0.25 and think budget 960 add nothing on A 001's weights).

## Next

- ~14:05: dm26 skip seed 2. Then decide on seeds 3-4 (full 6-h pipelines; worth it if seed 2
  leaves them needing <= ~0.15 each).
- ~14:20 and ~15:30: skip_cos seeds 3-4 (5-seed form; seed 3 >= 0.31 and > 0.2403 settles it alone).
  If met: commit `records/v3/track_a/002_<slug>/config.json` (draft in the scratchpad) and flag it.
- ~14:35 (seed 0) and ~19:30 (seeds 1-2): r6, think budget 768 in training. Every seed >= 0.2703
  and none collapsed would make it the A candidate.
- a01_skip AWS seeds 0-2 whenever the laptop commits them.

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
