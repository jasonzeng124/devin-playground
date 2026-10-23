# Status

(maintained by the research agent; see AGENT.md)

**Promotion (2026-10-23 14:25): track A, `calibration/research_r2/a01_fp32_lr1e-6_skip_cos` ->
record `records/v3/track_a/002_fp32_skip_cosine` (config.json committed).** In-loop final dev,
seeds 0-4: 0.300 / 0.270 / 0.472 / 0.315 / 0.263 = 972 of 3,000 (0.324). The 5-seed form needs 811
(A 001's record seeds: 721 of 3,000 = 0.2403; line 0.2703). All five are above 0.2403 and none
collapsed (seeds 0, 1, 3 and 4 are one-guess by the informative-groups count, all above the holder's
mean). Sampler: T 0.5, constrained, no think budget. T 0.5 was chosen on A 001's dev evals;
skip_cos's final evals have 0-0.2 % truncation, so a budget changes nothing. N = 5 (seeds 0-4),
`scripts/run_seed_v3.sh` STAGE=all, then
`validate_record_v3.py --compare-to records/v3/track_a/001_ladder_constrained`. Engaging's only
mit_normal_gpu H100 node (node2906) estimates its next start at 10-26 06:46, so AWS via the laptop
is likely sooner. All 5 seeds on one provider. I have not submitted Engaging record workers.

Updated 2026-10-23 19:40 EDT.

## Running

- Workers: 6 on mit_preemptable (`RUNNER=scripts/research/run_item.sh`, RESERVE_MIN 330, H100 80GB
  HBM3; node1800 and the NVL nodes excluded). The QOS allows 4 GPUs per user; the partition is
  busy, so 2 run:
  - 26644473 (`main:long`, node2700): r7 `a01_fp32_lr1e-6_skip_cos_mal0_s300` seed 3 (18:30 -> ~20:45).
  - 26644472 (`main:long`, node2700): r7 seed 4 (18:52 -> ~21:10).
- Pending (Priority): 26644470-71 (preempted on node2701 at 14:59), 26644474 (preempted on node3009
  at 17:37, 90 s after starting), 26644475. 26644470 redoes c90_cc1e-3 seed 0 when it starts; the
  others pop fresh items.
- Successors 26649456-59, each `afterany` on one of 26644470-73, same queues.
- AWS (laptop session): a01_skip seeds 0-2 on on-demand p5.4xlarge (run a01_skip-aws-od-20261022),
  dev only; results not here yet.

## Queues (~75 GPU-hours)

- `main` (17): `dm26_fp32_lr3e-6_skip_pipe` seeds 3-4 (6-h pipelines); r7 sampler evals for seeds 0
  and 2 (T 0.5, no budget; ~30 min each); r8 `a01_fp32_lr1e-6_skip_cos_mal0_until300` seeds 0-2; r4 dm51_fp32_lr3e-6_skip seeds 0-2;
  d00_fp32_lr3e-6_skip seeds 0-2; a01_skip seeds 3-4; d00_fp32_lr3e-6_skip seeds 3-4.
- `long` (1): c90_pos01 seed 0 (c90_cc1e-3 seed 0 is in 26644470's hands).
- `later` (14, read by no worker): 11 c90 seeds; r6 seeds 1-2 and its tb768 eval (parked 10-23).

## Best so far

- Track A: skip + cosine (above), promoted. r7 = r5 (malformed reward 0) stopped at step 300; r5's
  seeds were 0.380 / 0.355 / 0.373 there, before collapsing after steps 350-550. r7 seeds 0-2:
  **0.385 / 0.348 / 0.238** (seed 2 fell from 0.372 at step 250: entropy collapse, not truncation).
  The 3-seed form is missed. The 5-seed form needs seeds 3-4 to total >= 228 puzzles and both to
  finish above 0.2403. Record sampler on seed 1 (T 0.5): dev16 0.375 without a budget, 0.376 with
  tb960, 0.365 with tb768, so r7's sampler is T 0.5 without a budget, as A 002's. r6 (think budget 768 in training) seed 0 ended at 0.318,
  healthy but slow (0.255 at step 300); parked. r8 = r5 with the -0.1 truncation penalty back from
  step 300 on (new option `malformed_reward_until`), full budget, queued.
- Track D: `dm26_fp32_lr3e-6_skip` seeds 0-2 at **0.152 / 0.143 / 0.153** (269 puzzles, mean 0.1494).
  The 3-seed form is out (seed 1); the 5-seed form needs 447 puzzles over 5 seeds, so seeds 3-4 must
  total >= 178 (0.148 each on average). They run as full pipelines (pretraining included).
- Holders: track A 001 (test 0.2481, dev 0.2403), track D 000 (test 0.1166, dev 0.1190). Promotion
  needs dev mean >= 0.2703 (A; 163 of 600) / 0.149 (D; 90 of 600), with 4 of 5 seeds above the
  holder's dev mean and at most 1 collapsed, or 3 seeds all that high and none collapsed. Once A 002
  lands, later A candidates are measured against its dev mean.
  "Collapsed" = diverged, or one-guess below the holder's dev mean (clarified 10-22, PLAN.md;
  QUESTIONS.md asks). On answer-constrained arms "diverged" only means truncated searches, so it
  counts like one-guess (clarified 10-23 07:00, before r5 ended; QUESTIONS.md asks).

## Next

- ~20:45 and ~21:10: r7 seeds 3-4 end (5-seed form: both above 0.2403 and >= 228 puzzles
  together). If met, r7 is flagged here (sampler T 0.5, constrained, no budget). If A 002 has landed
  by then, r7 is measured against A 002.
- dm26 pipeline seeds 3-4 start after those (~21:00-22:00); check pretraining tokens/s against ~370k.
  They end ~6 h after they start (D 5-seed form).
- a01_skip AWS seeds 0-2 whenever the laptop commits them.

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q`: without it the two-rank tests in
  `tests/test_distributed.py` fail on the login node ("libgomp: Thread creation failed").
- Eval arms: a dir with `eval.json` queued like a screen runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`, `dev_ladder_eval.py`); `summarize_screens.py` prints them.
- Screens that save weights write their logs to the pool weights dir
  (`~/orcd/pool/rlvr-ckpts/<arm>/seeds/<seed>/`) while they run; the repo seed dir gets them at the end.
- Track D pretraining speed varies by node: 224k tokens/s on node4109 (4 of our jobs at once)
  against 372k on node3009 and 359-373k in D 000's record. Pretraining is time-budgeted, so D2
  results must be read with their tokens/s. The 6 saved D2 models all ran at ~370k; check the dm26
  pipeline seeds' tokens/s.
- Ubuntu 24 nodes (8 of ~35 H100 nodes on mit_preemptable): gcc needs the multiarch paths, which
  the worker now adds. `module load` fails there because lmod isn't installed (probe 26644751 on
  node2801), so `cuda/13.0.1` isn't loaded; vLLM and training start without it.
