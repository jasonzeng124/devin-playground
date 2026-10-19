# Status

(maintained by the research agent; see AGENT.md)

Updated 2026-10-19 15:40 EDT.

## Running

- **r1** (`calibration/research_r1`, queue `r1`; fp32 master weights + lr sweep for A 001's recipe
  with and without the constraint, and for D 000's RL): 6 seed-0 screens on 6 GPUs (2
  mit_normal_gpu, 4 preemptable), 19 queued. Done so far: none of r1 itself; `d00_fp32` (w42)
  finished (LOG 2026-10-19 evening).
- Preemption: five preemptable restarts today. Since a1e4502, screens on mit_preemptable save
  resumable state hourly (`$CKPT_DIR/<arm>/resume/<seed>/`, deleted when the seed finishes); the
  a01_fp32_lr1e-6 seed 0 rerun (job 26360542, from 15:36 EDT) is the first to use it.
- Workers: the 4 preemptable workers end at the maintenance start (00:00 EDT); mit_normal_gpu
  workers run one item each (4 more pending, 2 at a time); 4 preemptable successors (26364066-69,
  `afterany` on the current 4) serve `r1:main` with `RUNNER=scripts/research/run_item.sh`.
- Queued after r1 (`queue/main.txt`): **r2 track D data screens** (D2), 3 seeds each:
  `calibration/research_r2/d00_dm26` and `d00_dm51` = D 000 pretrained on FineWeb-Edu mixed with
  DeepMind mathematics_dataset arithmetic (26 % / 51 % of tokens; data in
  `~/orcd/pool/data/dm_math`). They pretrain their own models (`run_seed_v3.sh STAGE=train` via
  `run_item.sh`; dev ladder only; no resume, so preemption restarts them).
- Engaging maintenance 2026-10-20 00:00-21:00 EDT: nothing runs.

## Best so far

No new candidate. Holders: track A 001 (test 0.2481, dev 0.240), track D 000 (test 0.1166, dev
0.119). Promotion needs dev mean >= 0.270 (A) / 0.149 (D) with at least 4 of 5 seeds above the
holder's dev mean (PLAN.md).

## Next

- 2026-10-19 ~16:40 EDT: check that the a01_fp32_lr1e-6 rerun saved its first resume state.
- 2026-10-19 ~20:30-21:00 EDT: the six seed-0 screens finish; LOG them.
- Before the successors reach r1's seed 1-2 items (after the maintenance, 2026-10-20 21:00 EDT):
  prune the arms whose seed 0 failed and queue r2 for track A in `queue/main.txt` (ahead of the
  pruned items).

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q`: without it the two-rank tests in
  `tests/test_distributed.py` fail on the login node ("libgomp: Thread creation failed").
- Eval arms: a dir with `eval.json` queued like a screen runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`, `dev_ladder_eval.py`); `summarize_screens.py` prints them.
