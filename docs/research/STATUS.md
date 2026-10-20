# Status

(maintained by the research agent; see AGENT.md)

**Promotion: none yet.** (When an arm meets the promotion rule, this line names the arm, its
config path and its dev numbers, and the record folder's config.json is committed; the record
then runs on AWS via the laptop session or on Engaging, whichever starts sooner.)

Updated 2026-10-20 18:25 EDT.

## Running

- Nothing runs yet: Engaging is crowded after its maintenance (ended ~13:45 EDT). The 4
  preemptable workers started at 13:46 and were preempted after 15-17 min (one restarted at 14:07,
  preempted after 6 min); each redoes its item when it restarts (`queue/running/<job>`).
- Workers: 6 preemptable (26424658-61 and, since 17:03, 26450849-50; 2-day limit), all pending on
  priority. 26424658-59 take `long` first, the others `r1` then `main`. The 6 mit_normal_gpu workers
  (26424662-67, `r1:main`, one item each) now wait for 26450849-50 to end (`afterany`). Slurm put
  their first start at 2026-10-23 07:30 (one 4-GPU H100 node in that partition), so their two
  slots go to preemptable workers until then, and the 6-GPU cap holds. All use
  `RUNNER=scripts/research/run_item.sh`.
- Queues, in order (in-flight items are in `queue/running/`):
  - `r1`: track D sampler evals first (~1 min each: the fp32 screen's seed 0 weights at T 1.0 /
    0.5 / 0.25, D 000's seeds at T 0.25), so a D record's sampler is chosen on its own weights; then
    d00_fp32_lr3e-6 seeds 3-4 (seeds 1-2 in flight; a 5-seed D screen with seed 0).
  - `main` (15, r2): a01_skip seeds 0-2 and a01_fp32_lr1e-6_skip_cos seed 0 first (one wave, so
    a01_skip's 3-seed promotion check comes ~5.5 h sooner), then that arm's seeds 1-2 and
    d00_fp32_lr3e-6_skip seeds 0-2, then seeds 3-4 of each, dropped if seeds 0-2 rule an arm out.
  - `long` (preemptable workers only; 30): d00_dm26 / d00_dm51 seed 0 (in flight), then r3 collapse
    screens (`calibration/research_r3/c90_*`, 90 min: malformed reward 0, T 1.2, Clip-Cov,
    positive advantages x 0.1) seeds 0-1, D2 seeds 1, collapse seeds 2-3, D2 seeds 2, then sampler
    evals on A 001's weights (T 0.5 + think budget 960; T 0.25).
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

- Slurm's estimate (18:20, with the workers' 8 CPUs / 64 GB) for the preemptable workers: ~01:55
  tonight (2,652 jobs pending). If all 6 start then: d00_fp32_lr3e-6 seeds 1-4 ~06:30 (the D
  promotion decision), D2 seed 0s ~08:10, first collapse screens ~10:00, a01_skip seeds 0-2 ~12:00
  (its 3-seed promotion check).
- Open question (QUESTIONS.md): may AWS run a01_skip's dev screens while Engaging is congested?
- When d00_fp32_lr3e-6 seeds 1-4 are in: if seeds 0-4 meet the promotion rule, commit the D
  record config (D 000's pretraining + fp32 RL at lr 3e-6, sampler T 0.5) and flag it above.
- About 6 h after the long workers start: first D2 pretraining screens (dm26 / dm51 seed 0); then
  the collapse screens (~1.7 h each). A collapse fix that keeps 4 of 4 seeds healthy at step 300
  (length >= 100 tokens, informative >= 15) goes into a full 5-h screen with skip + cosine.
- Before ~2026-10-22 evening: successors for the first 4 preemptable workers (2-day limit from
  their start); the mit_normal_gpu workers follow 26450849-50.

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q`: without it the two-rank tests in
  `tests/test_distributed.py` fail on the login node ("libgomp: Thread creation failed").
- Eval arms: a dir with `eval.json` queued like a screen runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`, `dev_ladder_eval.py`); `summarize_screens.py` prints them.
- Track D pretraining speed varies by node: 224k tokens/s on node4109 (4 of our jobs at once)
  against 372k on node3009 and 359-373k in D 000's record. Pretraining is time-budgeted, so D2
  results must be read with their tokens/s.
