# Status

(maintained by the research agent; see AGENT.md)

**Promotion: none yet.** (When an arm meets the promotion rule, this line names the arm, its
config path and its dev numbers, and the record folder's config.json is committed; the record
then runs on AWS via the laptop session or on Engaging, whichever starts sooner.)

Updated 2026-10-20 16:00 EDT.

## Running

- Engaging's maintenance ended early (~13:45 EDT). The 4 preemptable workers started at 13:46,
  were all preempted within 15-25 min and requeued; each redoes its item when it restarts
  (`queue/running/<job>`). All 10 workers are pending on priority in the post-maintenance rush:
  4 preemptable (26424658-61; 2 take `long` first, 2 take `r1` then `main`; 2-day limit) and 6
  mit_normal_gpu (26424662-67, `r1:main`, one item each, 2 at a time). All use
  `RUNNER=scripts/research/run_item.sh`.
- Queues, in order (in-flight items are in `queue/running/`):
  - `r1`: d00_fp32_lr3e-6 seeds 1-2 (in flight), then seeds 3-4 (5-seed D screen with seed 0).
  - `main` (9, r2): a01_skip, a01_fp32_lr1e-6_skip_cos, d00_fp32_lr3e-6_skip (seeds 0-2, interleaved).
  - `long` (preemptable workers only; 31): d00_dm26 / d00_dm51 seed 0 (in flight), then r3 collapse
    screens (`calibration/research_r3/c90_*`, 90 min) seeds 0-1, D2 seeds 1, collapse seeds 2-3,
    D2 seeds 2, then sampler evals on the holders' weights (A 001 T 0.5 + think budget 960,
    A 001 T 0.25, D 000 T 0.25).
  - Dropped 2026-10-20 (LOG): a01_fp32_lr1e-6 seeds 1-2, r2's `_cos` / `_mal0` arms.
- Preemption: screens on mit_preemptable save resumable state hourly (a1e4502). D2 items
  (`run_seed_v3.sh`) do not resume.

## Best so far

No promotion yet. Leads (one seed each): track D fp32 lr 3e-6 RL, 0.147 vs D 000's 0.087 on the
same pretrained seed; track A fp32 lr 1e-6, 0.290 vs A 001's seed-0 0.203 (its first attempt went
one-guess). Holders: track A 001 (test 0.2481, dev 0.240), track D 000 (test 0.1166, dev 0.119).
Promotion needs dev mean >= 0.270 (A) / 0.149 (D) with at least 4 of 5 seeds above the holder's
dev mean, or 3 seeds all at least that high and none collapsed (PLAN.md). Sampler: T 0.5 beats
T 1.0 on all 10 holder seeds (+0.0085 A, +0.0024 D); record attempts use T 0.5.

## Next

- Slurm's estimates (16:00): preemptable H100 jobs start ~21:20 tonight; the mit_normal_gpu
  workers from 2026-10-21 03:30 (that partition has one 4-GPU H100 node; new H100 or H200 jobs
  there would wait until ~10-22 15:30). Expected: d00_fp32_lr3e-6 seeds 1-2 ~01:30, seeds 3-4
  ~05:30, D2 seed 0s ~03:30, first collapse screens ~05:00-08:00, first r2 A screens ~09:00-10:30.
- About 4 h after the r1 workers start: d00_fp32_lr3e-6 seeds 1-2. If seeds 0-4 meet the promotion
  rule, commit the D record config (D 000's pretraining + fp32 RL at lr 3e-6) and flag it above.
- About 6 h after the long workers start: first D2 pretraining screens (dm26 / dm51 seed 0); then
  the collapse screens (~1.7 h each). A collapse fix that keeps 4 of 4 seeds healthy at step 300
  (length >= 100 tokens, informative >= 15) goes into a full 5-h screen with skip + cosine.
- Before ~2026-10-22 evening: successors for the preemptable workers (2-day limit from their start).

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q`: without it the two-rank tests in
  `tests/test_distributed.py` fail on the login node ("libgomp: Thread creation failed").
- Eval arms: a dir with `eval.json` queued like a screen runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`, `dev_ladder_eval.py`); `summarize_screens.py` prints them.
- Track D pretraining speed varies by node: 224k tokens/s on node4109 (4 of our jobs at once)
  against 372k on node3009 and 359-373k in D 000's record. Pretraining is time-budgeted, so D2
  results must be read with their tokens/s.
