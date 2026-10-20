# Status

(maintained by the research agent; see AGENT.md)

Updated 2026-10-19 21:00 EDT.

## Running

- Engaging maintenance 2026-10-20 00:00-21:00 EDT: nothing runs. If the login node reboots, this
  agent's tmux session ends; restart it with `scripts/research/start_agent.sh`.
- Until 23:59 tonight: 3 short eval workers (queue `evals`) finish the sampler-temperature check
  (A 001 and D 000 record seeds on the dev ladder at T 0.5 / 0.75 / 1.0; `calibration/research_r2/samp_*`).
- From 2026-10-20 21:00 EDT: 4 preemptable workers (26424658-59 take `long` first, 26424660-61 take
  `r1` then `main`; 2-day limit, so they end ~2026-10-22 21:00) and 6 mit_normal_gpu workers
  (26424662-67, `r1:main`, one item each, 2 at a time). All use `RUNNER=scripts/research/run_item.sh`.
- Queues, in order:
  - `r1` (6): a01_fp32_lr1e-6 seeds 1-2; d00_fp32_lr3e-6 seeds 1-4 (5-seed D screen with seed 0).
  - `main` (15, r2): a01_skip, a01_fp32_lr1e-6_skip_cos, d00_fp32_lr3e-6_skip (seeds 0-2, interleaved),
    then a01_fp32_lr1e-6_cos and _mal0 (seeds 0-2).
  - `long` (6, D2): d00_dm26 / d00_dm51 seeds 0-2 (pretraining on FineWeb-Edu + DeepMind arithmetic, 6 h each).
- Preemption: screens on mit_preemptable save resumable state hourly (a1e4502; checked on a real
  fp32 run). D2 items (`run_seed_v3.sh`) do not resume.

## Best so far

No promotion yet. Leads (one seed each): track D fp32 lr 3e-6 RL, 0.147 vs D 000's 0.087 on the
same pretrained seed; track A fp32 lr 1e-6, 0.290 vs A 001's seed-0 0.203 (its first attempt went
one-guess). Holders: track A 001 (test 0.2481, dev 0.240), track D 000 (test 0.1166, dev 0.119).
Promotion needs dev mean >= 0.270 (A) / 0.149 (D) with at least 4 of 5 seeds above the holder's
dev mean, or 3 seeds all at least that high and none collapsed (PLAN.md).

## Next

- Tonight ~21:15 EDT: the sampler evals finish; LOG them.
- 2026-10-21 ~01:00-03:00 EDT: first post-maintenance results (d00_fp32_lr3e-6 seeds 1-2,
  a01_fp32_lr1e-6 seeds 1-2). If d00_fp32_lr3e-6 reaches the promotion rule, prepare the D
  record (D 000's pretraining + fp32 RL at lr 3e-6; 5 seeds of the whole pipeline).
- Before ~2026-10-22 21:00: successors for the preemptable workers.

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q`: without it the two-rank tests in
  `tests/test_distributed.py` fail on the login node ("libgomp: Thread creation failed").
- Eval arms: a dir with `eval.json` queued like a screen runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`, `dev_ladder_eval.py`); `summarize_screens.py` prints them.
