# Status

(maintained by the research agent; see AGENT.md)

**Promotion: none yet.** (When an arm meets the promotion rule, this line names the arm, its
config path and its dev numbers, and the record folder's config.json is committed; the record
then runs on AWS via the laptop session or on Engaging, whichever starts sooner.)

Updated 2026-10-21 00:45 EDT.

## Running

- AWS (laptop session, maintainer-approved): a01_skip seeds 0-2 on spot p5.4xlarge (1 H100
  SXM, us-east-1; run `a01_skip-aws-20261021`, launched ~23:20), dev ladder only (STAGE=train).
  The laptop session commits the results into `calibration/research_r2/a01_skip/seeds/`.
- Engaging: nothing runs yet; it is crowded after its maintenance (ended ~13:45 EDT). The 4
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
  - `main` (12, r2): a01_fp32_lr1e-6_skip_cos seeds 0-2 and d00_fp32_lr3e-6_skip seed 0 first,
    then d00_fp32_lr3e-6_skip seeds 1-2 with a01_skip seeds 3-4 (seeds 0-2 are on AWS), then seeds
    3-4 of the other two arms. Seeds 3-4 are dropped if seeds 0-2 rule an arm out, or are not needed
    because the arm promoted on 3 seeds.
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

- Slurm's estimate for the preemptable workers keeps slipping as the queue grows: ~01:55 at 18:20
  (2,652 jobs pending), ~05:20 on 10-21 at 22:25 (2,925 pending; age adds no priority on
  mit_preemptable), and none at all at 00:30 (3,071 pending). If all 6 start ~05:20: d00_fp32_lr3e-6 seeds 1-4 ~10:15 (the D promotion
  decision), D2 seed 0s ~11:40, first collapse screens ~13:20, first `main` wave ~10:00-15:30.
- a01_skip seeds 0-2 from AWS (~05:00-06:00 on 10-21 if the spot instances hold): the 3-seed
  promotion check (every seed healthy and >= 0.270). If it passes: commit the A record config
  (`records/v3/track_a/002_<slug>/`), flag it above, and take seeds 3-4 out of `main`. If one seed
  misses or collapses, seeds 3-4 move to the front of `main` for the 5-seed form.
- QUESTIONS.md (00:45) asks whether d00_fp32_lr3e-6 seeds 1-4 may run on AWS as full-pipeline
  dev screens (`calibration/research_r2/d00_fp32_lr3e-6_full`, ~25 instance-hours), and a01_skip
  seeds 3-4 too if the 5-seed form needs them. If yes, those seeds leave the Engaging queues.
- When d00_fp32_lr3e-6 seeds 1-4 are in: if seeds 0-4 meet the promotion rule, commit the D
  record config (D 000's pretraining + fp32 RL at lr 3e-6, with the sampler temperature the D
  sampler evals pick) and flag it above. Also check whether their RL levels off early, as seed 0's
  did (flat from ~75 of 234 min). If so, queue a split screen that moves an hour from RL to
  pretraining (172 + 174 min).
- About 6 h after the long workers start: first D2 pretraining screens (dm26 / dm51 seed 0); then
  the collapse screens (~1.7 h each). A collapse fix that keeps 4 of 4 seeds healthy at step 300
  (length >= 100 tokens, informative >= 15) goes into a full 5-h screen with skip + cosine.
- About 2 days after the workers start (~2026-10-23 morning if they start ~05:20 on 10-21):
  successors for the preemptable workers; the mit_normal_gpu workers follow 26450849-50.
  At 22:25 Claude Code's auto mode refused an `sbatch` as a shared-cluster change. It was an
  eval-only worker for the idle H100 NVL node1708, with 26450850 to be chained behind it, and
  nothing was submitted. If successor or record submissions are refused too, the maintainer or
  the laptop session has to submit them, or allow `sbatch`/`scontrol` for this session.

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q`: without it the two-rank tests in
  `tests/test_distributed.py` fail on the login node ("libgomp: Thread creation failed").
- Eval arms: a dir with `eval.json` queued like a screen runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`, `dev_ladder_eval.py`); `summarize_screens.py` prints them.
- Track D pretraining speed varies by node: 224k tokens/s on node4109 (4 of our jobs at once)
  against 372k on node3009 and 359-373k in D 000's record. Pretraining is time-budgeted, so D2
  results must be read with their tokens/s.
