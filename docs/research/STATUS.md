# Status

(maintained by the research agent; see AGENT.md)

**Track D record D 002 is queued on Engaging (promoted 2026-10-24 02:35): `dm26_fp32_lr3e-6_skip`
(+ `_pipe` seeds 3-4) -> `records/v3/track_d/002_dm_math_fp32` (config.json committed).** In-loop
final dev, seeds 0-4: 0.152 / 0.143 / 0.153 / 0.158 / 0.157 = 458 of 3,000 (0.153; D 000's line
0.149 = 447). D 000's pretraining with DeepMind mathematics_dataset arithmetic (published v1.0,
as published) as ~26 % of the tokens, then fp32 master weights at lr 3e-6 with skipping. Sampler
T 1.0, constrained, no budget. N = 5 on the mit_preemptable screen workers (H100 80GB HBM3) through
`main` (`run_item.sh` runs `records/v3/` items with STAGE=all `run_seed_v3.sh`). **Test ladder so
far: seed 0 0.1496 (finished 09:25), seed 2 0.1361 (13:04)**, each using 348 of 360 min. D 000's
seeds score 0.098-0.139, mean 0.1166. Seed 3 (started 09:25) ends ~15:15 and seed 4 (started 13:04)
~18:50. Seed 1 was preempted 2 min before the end of its pretraining; it reruns when its worker
restarts. Don't start it on AWS.

**Two track A record attempts are waiting, both measured against A 001 (dev 0.2403, line 0.2703 =
811 of 3,000 puzzles) because neither has landed. Whichever lands first becomes the holder, and the
other would then need 0.354, which neither reaches on dev. So start both before either lands, or
pick one.** All seeds of a record on one provider; N = 5 (seeds 0-4), `scripts/run_seed_v3.sh`
STAGE=all, then `validate_record_v3.py --compare-to records/v3/track_a/001_ladder_constrained`.

- **A 003 (promoted 2026-10-23 21:10): `calibration/research_r7/a01_fp32_lr1e-6_skip_cos_mal0_s300`
  -> `records/v3/track_a/003_fp32_mal0_s300` (config.json committed).** In-loop final dev, seeds
  0-4: 0.385 / 0.348 / 0.238 / 0.348 / 0.302 = 973 of 3,000 (0.324, median 0.348). Four seeds are
  above 0.2403 and none collapsed (seed 3's "diverged" label is 60.5 % truncated greedy outputs at
  0.348). Training reward 0 for truncated outputs; training stops at step 300 (115-143 min), so a
  seed costs under half of A 002's GPU time. Sampler: T 0.5, constrained, no think budget: dev16
  0.399 / 0.375 / 0.262 / 0.357 / 0.292 = 0.337, +0.013 over greedy (A 001: +0.015). On seed 1 a
  think budget of 960 adds 0.001 and 768 costs 0.010.
- **A 002 (promoted 2026-10-23 14:25): `calibration/research_r2/a01_fp32_lr1e-6_skip_cos` ->
  `records/v3/track_a/002_fp32_skip_cosine` (config.json committed).** In-loop final dev, seeds
  0-4: 0.300 / 0.270 / 0.472 / 0.315 / 0.263 = 972 of 3,000 (0.324, median 0.300). All five are
  above 0.2403 and none collapsed. Sampler: T 0.5, constrained, no think budget (0-0.2 %
  truncation, so a budget changes nothing). Trains the full 290 min.
- **If only one runs, run A 002 (changed 10-24 00:55).** r8 seed 0 ran A 003's exact training
  through step 299 (r8 changes the reward only from step 300 on) and collapsed at steps 264-284
  (dev 0.363 at step 250, 0.053 at step 300). Over the nine runs of A 003's recipe through step 300
  (r5's three, r7's five, r8 seed 0), step 300 averages 0.309, below A 002's 0.324. Two of the
  nine fell between steps 250 and 300, so about one record seed in five may collapse. A 002's five
  seeds all ran the full 290 min without collapsing. A 003 still meets the rule, and it costs half
  as much. The record sampler adds about as much to r7 as to A 001 (+0.013), so it doesn't separate
  the two.
- Engaging: node2906, the only mit_normal_gpu H100 80GB HBM3 node, estimates 10-26 15:07 for a
  6-h job, and my six preemptable workers fill the 6-GPU cap, so I have not submitted Engaging
  record workers. AWS via the laptop is sooner.

Updated 2026-10-24 13:15 EDT.

## Running

- Workers: 6 on mit_preemptable (`RUNNER=scripts/research/run_item.sh`, RESERVE_MIN 330, H100 80GB
  HBM3; node1800 and the NVL nodes excluded). The QOS allows 4 GPUs per user; 3 run:
  - 26644472 (`main:long`, node2700): D 002 record seed 3 (09:25 -> ~15:15).
  - 26644473 (`main:long`, node2700): D 002 record seed 4 (13:04 -> ~18:50).
- Pending (Resources), each redoing its item in hand when it restarts:
  - 26644471: D 002 seed 1, preempted at 06:42 at 110 of 112 pretraining minutes. The rerun starts
    from scratch; the interrupted run goes to `extra/interrupted_1_1` and is reported.
  - 26644475: r8 seed 1, preempted at 03:55 (dev 0.390 at 124 min); resumes from its hourly
    checkpoint.
  - 26644474: r9 seed 0, preempted at 05:45, 28 s after it started.
  - 26644470: c90_cc1e-3 seed 0, preempted at 02:45.
- `run_item.sh` (10-24 04:55 and 07:00), for `records/v3/` items: a seed stopped during its data copy
  or pretraining has its partial run kept under `<record>/extra/interrupted_<seed>_<n>` before the
  rerun (the worker already did this for later stops). And a record seed isn't started with less
  job time left than its budgets + 30 min (376 min for D 002). Restarted workers get backfill limits
  as short as 5:30. A refused item shows FAILED in worker.log with a "not started, requeue it" note,
  and I requeue it by hand.
- Successors 26649456-59, each `afterany` on one of 26644470-73, same queues. 26644472-73 reach
  their 2-day limit at 10-25 00:26 and pop nothing after ~18:56 on 10-24.
- AWS (laptop session): a01_skip seeds 0-2 on on-demand p5.4xlarge (run a01_skip-aws-od-20261022),
  dev only; results not here yet.

## Queues (~110 GPU-hours)

- `main` (15): r10 `a01_fp32_lr1e-6_skip_cos_advnone` seeds 0-2 (A 002 with
  mean-centred advantages, queued 07:35), r9 `a01_fp32_lr1e-6_cos_mal0_cc1e-3` seeds 1-2 (r5 + Clip-Cov
  0.1 %, skipping off; seed 0 is in 26644474's hands) and r4 dm51_fp32_lr3e-6_skip seeds 0-2,
  interleaved; then d00_fp32_lr3e-6_skip seeds 0-2, a01_skip seeds 3-4, d00_fp32_lr3e-6_skip
  seeds 3-4.
- `long` (1): c90_pos01 seed 0 (c90_cc1e-3 seed 0 is in 26644470's hands).
- `later` (16, read by no worker): 11 c90 seeds; r6 seeds 1-2 and its tb768 eval (parked 10-23);
  r8 seeds 3-4 (parked 10-24 07:30).

## Best so far

- Track A: A 002 (skip + cosine) and A 003 (r7: r5's malformed reward 0, stopped at step 300), both
  promoted (top). r5, the same recipe with the full budget, had seeds at 0.380 / 0.355 / 0.373 at
  step 300 and collapsed after steps 350-550 as truncated searches took over. r7's seed 2 fell from
  0.372 at step 250 to 0.238 (entropy collapse, not truncation). r6 (think budget 768 in training)
  seed 0 ended at 0.318, healthy but slow; parked. r8 = r5 with the -0.1 truncation penalty back
  from step 300 on (new option `malformed_reward_until`), full budget: can the remaining 2.8 h add
  to r7's gains without the collapse? No: seed 0 collapsed before the switch (0.205), and seed 2
  peaked at 0.415 at step 500, then fell to 0.315 by the end as truncation crept back. Seed 1
  finishes when its worker restarts; seeds 3-4 parked.
- Track D: D 002 (dm26), promoted (top): 458 of 3,000 over five seeds (0.153). Record seeds 0 and 2
  scored 0.1496 and 0.1361 on the test ladder.
- Holders: track A 001 (test 0.2481, dev 0.2403), track D 000 (test 0.1166, dev 0.1190). Promotion
  needs dev mean >= 0.2703 (A; 163 of 600) / 0.149 (D; 90 of 600), with 4 of 5 seeds above the
  holder's dev mean and at most 1 collapsed, or 3 seeds all that high and none collapsed. Once A 002
  or A 003 lands, later A candidates are measured against its dev mean.
  "Collapsed" = diverged, or one-guess below the holder's dev mean (clarified 10-22, PLAN.md;
  QUESTIONS.md asks). On answer-constrained arms "diverged" only means truncated searches, so it
  counts like one-guess (clarified 10-23 07:00, before r5 ended; QUESTIONS.md asks). QUESTIONS.md
  also asks whether promotion should use dev under the record sampler (10-23 19:35).

## Next

- ~15:15: D 002 seed 3 ends, and 26644472 then starts r10 seed 0. ~18:50: seed 4 ends. Seed 1
  reruns when 26644471 restarts; after it, the README and `validate_record_v3.py`.
- r10 (A 002 + `adv_norm: none`): do the seeds keep searching past step 300 without collapsing
  (LOG 10-24 07:35)? Seed 0 starts after D 002 seed 4.
- No D 002 seed is left in `main` (seed 1 is in 26644471's hands), so the 10-25 00:26 time limit of
  26644472-73 no longer touches the record.
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
