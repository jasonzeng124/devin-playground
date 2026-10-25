# Status

(maintained by the research agent; see AGENT.md)

**Track D record D 002 landed (2026-10-24 22:39): `records/v3/track_d/002_dm_math_fp32` scores
0.1512 ± 0.0202 on the test ladder over five seeds, against D 000's 0.1166 (one-sided Welch
p = 0.0092). It is the track D holder and on the README.md leaderboard (LOG 22:39).** Its record
seeds' in-loop final dev is 0.167 / 0.137 / 0.150 / 0.193 / 0.155 = 481 of 3,000 (0.160), so the
new track D promotion line is 0.190 (571 of 3,000).

**Record workers are on HOLD (laptop, INBOX 2026-10-25 00:50): the maintainer is changing the
reference GPU to H200, with details to follow. Submit no record workers until then.** On the
laptop's earlier note (00:20) I submitted A 002's five record workers at 00:45. I cancelled them at
00:46, before any started (LOG 00:47). Their queue file is set aside as
`queue/record-002_fp32_skip_cosine.held`.

**Two track A record attempts are waiting, both measured against A 001 (dev 0.2403, line 0.2703 =
811 of 3,000 puzzles) because neither has landed. Whichever lands first becomes the holder, and the
other would then need 0.354, which neither reaches on dev. The laptop chose A 002 first and holds
A 003 until A 002 lands or the maintainer decides on an AWS capacity block (INBOX 10-25 00:20).**
All seeds of a record on one provider; N = 5 (seeds 0-4), `scripts/run_seed_v3.sh` STAGE=all,
then `validate_record_v3.py --compare-to records/v3/track_a/001_ladder_constrained`.

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
- Engaging: node2906, the only mit_normal_gpu H100 node (4 GPUs, HBM3), estimated 10-27 09:04 for
  a 6-h job, with 34 H100 jobs pending (checked 10-25 00:45). mit_normal_gpu also has 13 H200 nodes
  of 8 GPUs. AWS found no H100 capacity in 48 h and ran nothing, so records don't count on it.

Updated 2026-10-25 00:50 EDT.

## Running

- Workers on mit_preemptable (`RUNNER=scripts/research/run_item.sh`, RESERVE_MIN 330, H100 80GB
  HBM3; node1800, node2801 and the NVL nodes excluded). The QOS allows 4 GPUs per user. 1 runs and
  3 wait after preemptions:
  - 26786843 (node2700): dm51 seed 0, since 22:36.
  - 26786844: r9 seed 0, preempted 23:40 at 2 h 12 min; pending.
  - 26803673: dm51 seed 1, preempted 23:51; pending.
  - 26786872: r10 seed 1, preempted 23:51; pending.
  - The three pending ones resume from their hourly saves.
- One of node2801's GPUs fails every CUDA call ("busy or unavailable"). Between 15:05 and 16:31 it
  stopped four workers: 26644471, 26644474, 26644475 and 26649457 (LOG 16:35). Their failed items
  are requeued. At 20:23 it stopped 26649458, but the CUDA check put its item back (LOG 20:30).
  - The four running workers are the replacements (`main:long`, node2801 excluded). Between 17:06
    and 21:00 each was preempted within 1-4 min of starting, some several times.
  - Since 16:32, `run_item.sh` tries CUDA before each item and puts the item back if that fails
    with a CUDA error. A worker that lands on that GPU then exits without using anything up. Since
    18:00, a timeout or other failure only gets a note in worker.log (LOG 18:00).
- `run_item.sh` (10-24 04:55 and 07:00), for `records/v3/` items: a seed stopped during its data copy
  or pretraining has its partial run kept under `<record>/extra/interrupted_<seed>_<n>` before the
  rerun (the worker already did this for later stops). And a record seed isn't started with less
  job time left than its budgets + 30 min (376 min for D 002). Restarted workers get backfill limits
  as short as 5:30. A refused item shows FAILED in worker.log with a "not started, requeue it" note,
  and I requeue it by hand.
- Successors: none pending. 26649458 and 26649459 landed on node2801 on 10-24 and exited without
  using up their items. I cancelled 26649456 (pending, no item in hand) at 00:45 to make room for
  the record workers. I may update or cancel my own pending jobs without asking (INBOX 10-25).

## Queues (~80 GPU-hours, counting the items in workers' hands)

- In workers' hands: r4 dm51_fp32_lr3e-6_skip seeds 0-1, r9 `a01_fp32_lr1e-6_cos_mal0_cc1e-3`
  seed 0 (r5 + Clip-Cov 0.1 %, skipping off) and r10 `a01_fp32_lr1e-6_skip_cos_advnone` seed 1
  (A 002 with mean-centred advantages).
- `main` (14):
  - First, interleaved: r9 seed 1, r11 dm26_fp32_lr1e-5_skip seed 0, r10 seed 2, r11
    dm26_fp32_lr3e-5_skip seed 0, r9 seed 2, dm51 seed 2.
  - Then r8 seed 1, which resumes from its 03:50 state.
  - Then d00_fp32_lr3e-6_skip seeds 0-2 and r11 seeds 1-2 of both rates.
- `long` (1): c90_pos01 seed 0.
- `later` (20, read by no worker): 11 c90 seeds; r6 seeds 1-2 and its tb768 eval (parked 10-23);
  r8 seeds 3-4 (parked 10-24 07:30); d00_fp32_lr3e-6_skip seeds 3-4 (parked 10-24 22:43); a01_skip
  seeds 3-4 (parked 10-25 00:45).
- `record-002_fp32_skip_cosine.held`: A 002 seeds 0-4, set aside during the hold.

## Best so far

- Track A: A 002 (skip + cosine) and A 003 (r7: r5's malformed reward 0, stopped at step 300), both
  promoted (top). r5, the same recipe with the full budget, had seeds at 0.380 / 0.355 / 0.373 at
  step 300 and collapsed after steps 350-550 as truncated searches took over. r7's seed 2 fell from
  0.372 at step 250 to 0.238 (entropy collapse, not truncation). r6 (think budget 768 in training)
  seed 0 ended at 0.318, healthy but slow; parked. r8 = r5 with the -0.1 truncation penalty back
  from step 300 on (new option `malformed_reward_until`), full budget: can the remaining 2.8 h add
  to r7's gains without the collapse? No: seed 0 collapsed before the switch (0.205), and seed 2
  peaked at 0.415 at step 500, then fell to 0.315 by the end as truncation crept back. Seed 1
  finishes when its worker restarts; seeds 3-4 parked. r10 (A 002 with mean-centred advantages)
  seed 0 switched to searching around step 150, kept truncation at 0.02-0.11, and ended the full
  290 min at 0.340 (best 0.378), not collapsed (LOG 20:30). A 002's seed 0 ended at 0.300,
  one-guess.
- Track D: D 002 is the holder (test 0.1512; LOG 22:39). Its recipe averaged 0.153 on dev over
  five research screens and 0.160 over its five record seeds. No other D screen has reached the new
  line of 0.190.
- Holders: track A 001 (test 0.2481, dev 0.2403), track D 002 (test 0.1512, dev 0.1603). Promotion
  needs dev mean >= 0.2703 (A; 163 of 600) / 0.190 (D; 115 of 600), with 4 of 5 seeds above the
  holder's dev mean and at most 1 collapsed, or 3 seeds all that high and none collapsed. Once A 002
  or A 003 lands, later A candidates are measured against its dev mean.
  "Collapsed" = diverged, or one-guess below the holder's dev mean (clarified 10-22, PLAN.md;
  QUESTIONS.md asks). On answer-constrained arms "diverged" only means truncated searches, so it
  counts like one-guess (clarified 10-23 07:00, before r5 ended; QUESTIONS.md asks). QUESTIONS.md
  also asks whether promotion should use dev under the record sampler (10-23 19:35).

## Next

- Track D, against D 002's line of 0.190: r11 (D 002's RL at lr 1e-5 and 3e-5, paired with r4's
  seeds 0-2; LOG 22:43) and dm51 (51 % DeepMind share). Move r11's seeds 1-2 up if a seed 0 is
  clearly above its r4 pair (0.152).
- r10 (A 002 + `adv_norm: none`): seed 0 searched for the full run and ended at 0.340 (LOG 20:30).
  Do seeds 1 and 2 search too? The 3-seed form needs both at >= 0.2703, none collapsed.
- c90_cc1e-3 seed 0 ended at 0.162 in one-guess mode, with entropy rising to 2.25 (LOG 16:57).
  Clip-Cov alone doesn't produce search; r9 tests it together with r5's recipe.
- Records: wait for the laptop's H200 details (INBOX), then start A 002 as they say.

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
