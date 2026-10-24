# Status

(maintained by the research agent; see AGENT.md)

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
  seed costs under half of A 002's GPU time. Sampler: T 0.5, constrained, no think budget (seed 1:
  dev16 0.375 against greedy 0.348; a think budget of 960 adds 0.001, 768 costs 0.010). Sampler
  evals for seeds 0, 2, 3 and 4 are running first in `main` (results by ~23:00).
- **A 002 (promoted 2026-10-23 14:25): `calibration/research_r2/a01_fp32_lr1e-6_skip_cos` ->
  `records/v3/track_a/002_fp32_skip_cosine` (config.json committed).** In-loop final dev, seeds
  0-4: 0.300 / 0.270 / 0.472 / 0.315 / 0.263 = 972 of 3,000 (0.324, median 0.300). All five are
  above 0.2403 and none collapsed. Sampler: T 0.5, constrained, no think budget (0-0.2 %
  truncation, so a budget changes nothing). Trains the full 290 min.
- If only one runs, I'd pick A 003: half the cost, a higher median seed, and +0.027 from the
  record sampler on the seed measured (A 002's gain can't be measured; its screens didn't save
  weights).
- Engaging: node2906, the only mit_normal_gpu H100 80GB HBM3 node, estimates 10-26 15:07 for a
  6-h job, and my six preemptable workers fill the 6-GPU cap, so I have not submitted Engaging
  record workers. AWS via the laptop is sooner.

Updated 2026-10-23 21:15 EDT.

## Running

- Workers: 6 on mit_preemptable (`RUNNER=scripts/research/run_item.sh`, RESERVE_MIN 330, H100 80GB
  HBM3; node1800 and the NVL nodes excluded). The QOS allows 4 GPUs per user; the partition is
  busy, so 3 run:
  - 26644472 (`main:long`, node2700): r7 sampler eval, seed 0 (20:53 -> ~21:25).
  - 26644473 (`main:long`, node2700): `dm26_fp32_lr3e-6_skip_pipe` seed 4 (20:28 -> ~02:30;
    pretraining at 368k tokens/s).
  - 26644475 (`main:long`, node2642): `dm26_fp32_lr3e-6_skip_pipe` seed 3 (19:56 -> ~02:00;
    367k tokens/s).
- Pending (Priority): 26644470-71 (preempted on node2701 at 14:59), 26644474 (preempted on node3009
  at 17:37). 26644470 redoes c90_cc1e-3 seed 0 when it starts; the others pop fresh items.
- Successors 26649456-59, each `afterany` on one of 26644470-73, same queues. 26644472-73 reach
  their 2-day limit at 10-24 23:59 and start no item after ~18:30 that day.
- AWS (laptop session): a01_skip seeds 0-2 on on-demand p5.4xlarge (run a01_skip-aws-od-20261022),
  dev only; results not here yet.

## Queues (~85 GPU-hours)

- `main` (18): r7 sampler evals for seeds 2, 3 and 4 (T 0.5, no budget; ~30 min each); r8
  `a01_fp32_lr1e-6_skip_cos_mal0_until300` seeds 0-2; r4 dm51_fp32_lr3e-6_skip seeds 0-2;
  d00_fp32_lr3e-6_skip seeds 0-2; a01_skip seeds 3-4; d00_fp32_lr3e-6_skip seeds 3-4; r8 seeds 3-4
  (dropped if r8's seeds 0-2 fail).
- `long` (1): c90_pos01 seed 0 (c90_cc1e-3 seed 0 is in 26644470's hands).
- `later` (14, read by no worker): 11 c90 seeds; r6 seeds 1-2 and its tb768 eval (parked 10-23).

## Best so far

- Track A: A 002 (skip + cosine) and A 003 (r7: r5's malformed reward 0, stopped at step 300), both
  promoted (top). r5, the same recipe with the full budget, had seeds at 0.380 / 0.355 / 0.373 at
  step 300 and collapsed after steps 350-550 as truncated searches took over. r7's seed 2 fell from
  0.372 at step 250 to 0.238 (entropy collapse, not truncation). r6 (think budget 768 in training)
  seed 0 ended at 0.318, healthy but slow; parked. r8 = r5 with the -0.1 truncation penalty back
  from step 300 on (new option `malformed_reward_until`), full budget, next in `main`: can the
  remaining 2.8 h add to r7's gains without the collapse?
- Track D: `dm26_fp32_lr3e-6_skip` seeds 0-2 at **0.152 / 0.143 / 0.153** (269 puzzles, mean 0.1494).
  The 3-seed form is out (seed 1); the 5-seed form needs 447 puzzles over 5 seeds, so seeds 3-4 must
  total >= 178 (0.148 each on average). They run as full pipelines (pretraining included).
- Holders: track A 001 (test 0.2481, dev 0.2403), track D 000 (test 0.1166, dev 0.1190). Promotion
  needs dev mean >= 0.2703 (A; 163 of 600) / 0.149 (D; 90 of 600), with 4 of 5 seeds above the
  holder's dev mean and at most 1 collapsed, or 3 seeds all that high and none collapsed. Once A 002
  or A 003 lands, later A candidates are measured against its dev mean.
  "Collapsed" = diverged, or one-guess below the holder's dev mean (clarified 10-22, PLAN.md;
  QUESTIONS.md asks). On answer-constrained arms "diverged" only means truncated searches, so it
  counts like one-guess (clarified 10-23 07:00, before r5 ended; QUESTIONS.md asks). QUESTIONS.md
  also asks whether promotion should use dev under the record sampler (10-23 19:35).

## Next

- ~21:25-23:00: r7 sampler evals for seeds 0, 2, 3 and 4 (dev16 for A 003's five seeds).
- r8 seeds 0-2 start after them (~23:00), about 5 h each.
- ~02:00 and ~02:30: dm26 pipeline seeds 3-4 end (D 5-seed form: >= 178 puzzles together).
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
