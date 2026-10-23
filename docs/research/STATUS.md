# Status

(maintained by the research agent; see AGENT.md)

**Promotion (2026-10-23 14:25): track A, `calibration/research_r2/a01_fp32_lr1e-6_skip_cos` ->
record `records/v3/track_a/002_fp32_skip_cosine` (config.json committed).** In-loop final dev, seeds
0-3: 0.300 / 0.270 / 0.472 / 0.315 = 814 puzzles; the 5-seed form needs 811 (A 001's record seeds:
721 of 3,000 = 0.2403, line 0.2703), so it holds whatever seed 4 scores (0.263 at 238 min, ends
~15:20). All four are above 0.2403 and none collapsed (seeds 0, 1, 3 are one-guess by the
informative-groups count, all well above the holder's mean). Sampler: T 0.5, constrained, no think
budget (T 0.5 chosen on A 001's dev evals; skip_cos's final evals have 0-0.2 % truncation, so a
budget changes nothing). N = 5 (seeds 0-4), `scripts/run_seed_v3.sh` STAGE=all, then
`validate_record_v3.py --compare-to records/v3/track_a/001_ladder_constrained`. Engaging's only
mit_normal_gpu H100 node (node2906) estimates its next start at 10-26 06:46, so AWS via the laptop
is likely sooner; all 5 seeds on one provider. I have not submitted Engaging record workers.

Updated 2026-10-23 14:25 EDT.

## Running

- Workers: 6 on mit_preemptable (`RUNNER=scripts/research/run_item.sh`, RESERVE_MIN 330, H100 80GB
  HBM3; node1800 and the NVL nodes excluded). 4 run; 26644474-75 pend because the mit_preemptable QOS
  allows 4 GPUs per user, so they only start when a running worker ends.
  - 26644470 (`long:main`, node2701): r6 `a01_fp32_lr1e-6_skip_cos_tb768` seed 0 (09:35 -> ~14:35).
  - 26644471 (`main:long`, node2701): r7 `a01_fp32_lr1e-6_skip_cos_mal0_s300` seed 0 (13:51 -> ~16:05).
  - 26644472 (`main:long`, node2700): r7 seed 1 (14:15 -> ~16:30).
  - 26644473 (`main:long`, node2700): `a01_fp32_lr1e-6_skip_cos` seed 4 (10:18 -> ~15:20).
- Successors 26649456-59, each `afterany` on one running worker (26644470-73), same queues, so
  they start as the current workers end (~10-24 evening) and the cap holds.
- AWS (laptop session): a01_skip seeds 0-2 on on-demand p5.4xlarge (run a01_skip-aws-od-20261022),
  dev only; results not here yet.

## Queues (~69 GPU-hours)

- `main` (15): r7 seed 2; `dm26_fp32_lr3e-6_skip_pipe` seeds 3-4 (full 6-h pipelines); r6 seeds 1-2;
  r4 dm51_fp32_lr3e-6_skip seeds 0-2; d00_fp32_lr3e-6_skip seeds 0-2; a01_skip seeds 3-4;
  d00_fp32_lr3e-6_skip seeds 3-4.
- `long` (2): c90_cc1e-3 seed 0, c90_pos01 seed 0 (90-min collapse probes).
- `later` (11, read by no worker): the other c90 seeds, parked 10-23 04:45.

## Best so far

- Track A: skip + cosine (above), promoted. r7 = r5 (malformed reward 0) stopped at step 300: r5's
  seeds were 0.380 / 0.355 / 0.373 there, before collapsing after steps 350-550; seeds 0-1 run, seed 2
  next. r6 (think budget 768 in training) seed 0: 0.338-0.340 at 249-281 min, healthy (r5 had
  collapsed by then).
- Track D: `dm26_fp32_lr3e-6_skip` seeds 0-2 at **0.152 / 0.143 / 0.153** (269 puzzles, mean 0.1494).
  The 3-seed form is out (seed 1); the 5-seed form needs 447 puzzles over 5 seeds, so seeds 3-4 must
  total >= 178 (0.148 each on average). They run as full pipelines (pretraining included) after r7.
- Holders: track A 001 (test 0.2481, dev 0.2403), track D 000 (test 0.1166, dev 0.1190). Promotion
  needs dev mean >= 0.2703 (A; 163 of 600) / 0.149 (D; 90 of 600), with 4 of 5 seeds above the
  holder's dev mean and at most 1 collapsed, or 3 seeds all that high and none collapsed. Once A 002
  lands, later A candidates are measured against its dev mean.
  "Collapsed" = diverged, or one-guess below the holder's dev mean (clarified 10-22, PLAN.md;
  QUESTIONS.md asks). On answer-constrained arms "diverged" only means truncated searches, so it
  counts like one-guess (clarified 10-23 07:00, before r5 ended; QUESTIONS.md asks).

## Next

- ~14:35: r6 seed 0 ends. Its seeds 1-2 are queued behind the dm26 pipelines.
- ~15:20: skip + cosine seed 4 ends (its value no longer changes the promotion).
- ~16:05, ~16:30, ~17:35: r7 seeds 0-2 end. Queue `calibration/research_r7/samp_s300_t050`, `_tb960`,
  `_tb768` (dev sampler evals) as each seed's weights appear. Every seed >= 0.2703 and none collapsed
  is the 3-seed form against A 001; if A 002 has landed by then, r7 is measured against A 002.
- ~22:00 and ~22:30: dm26 pipeline seeds 3-4 end (D 5-seed form).
- a01_skip AWS seeds 0-2 whenever the laptop commits them.

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q`: without it the two-rank tests in
  `tests/test_distributed.py` fail on the login node ("libgomp: Thread creation failed").
- Eval arms: a dir with `eval.json` queued like a screen runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`, `dev_ladder_eval.py`); `summarize_screens.py` prints them.
- Track D pretraining speed varies by node: 224k tokens/s on node4109 (4 of our jobs at once)
  against 372k on node3009 and 359-373k in D 000's record. Pretraining is time-budgeted, so D2
  results must be read with their tokens/s. The 6 saved D2 models all ran at ~370k; check the dm26
  pipeline seeds' tokens/s.
- Ubuntu 24 nodes (8 of ~35 H100 nodes on mit_preemptable): gcc needs the multiarch paths, which
  the worker now adds. `module load` fails there because lmod isn't installed (probe 26644751 on
  node2801), so `cuda/13.0.1` isn't loaded; vLLM and training start without it.
