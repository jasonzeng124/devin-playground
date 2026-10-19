# Status

(maintained by the research agent; see AGENT.md)

Updated 2026-10-19 12:05 EDT.

## Running

- **r1** (`calibration/research_r1`, queue `r1`; 26 dev-ladder screens, ~120 GPU-h): fp32 master
  weights + lr sweep for A 001's recipe (with and without the constraint) and D 000's RL, plus
  the w42 rerun of d00_fp32 seed 0. 3 running, 23 queued.
- Workers (at most 6 GPUs at once): 3 preemptable running (they end at the maintenance start),
  1 preemptable + 6 mit_normal_gpu pending (2 run at a time), and 4 preemptable successors
  (26364066-69, each `afterany` on one of the 4 preemptable workers) that serve `r1:main` with
  `RUNNER=scripts/research/run_item.sh`, so they continue with `queue/main.txt` when r1 drains.
- Queued after r1 (`queue/main.txt`): **r2 track D data screens** (D2), 3 seeds each:
  `calibration/research_r2/d00_dm26` and `d00_dm51` = D 000 with its pretraining tokens mixed with
  DeepMind mathematics_dataset arithmetic (26 % / 51 % of tokens; 0.299B unique math tokens in
  `~/orcd/pool/data/dm_math`, prepared by CPU job 26365155 in 3 min). They pretrain their own
  models (`run_seed_v3.sh STAGE=train` via `run_item.sh`; dev ladder only).
- Engaging maintenance 2026-10-20 00:00-21:00 EDT: nothing runs. r1 should finish around
  2026-10-21 evening.

## Best so far

No new candidate. Holders: track A 001 (test 0.2481, dev 0.240), track D 000 (test 0.1166, dev
0.119). Promotion needs dev mean >= 0.270 (A) / 0.149 (D) with at least 4 of 5 seeds above the
holder's dev mean (PLAN.md).

## Next

- Today ~15:40-16:10 EDT: the first r1 seeds finish (d00_fp32 s0, a01_fp32_lr1e-6 s0,
  d00_fp32_lr3e-6 s0); read their entropy/length curves.
- Before r1 drains (~2026-10-21): queue r2 in `queue/main.txt` from r1's results (A2 stability on
  the best fp32 lr; a dev-ladder sampler-temperature sweep of the A 001 seeds as eval arms).
- After the data job: track D pretraining screens on the DM-math mixes (D2).

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q`: without it the two-rank tests in
  `tests/test_distributed.py` fail on the login node ("libgomp: Thread creation failed").
- Eval arms: a dir with `eval.json` queued like a screen runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`, `dev_ladder_eval.py`); `summarize_screens.py` prints them.
