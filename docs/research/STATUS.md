# Status

(maintained by the research agent; see AGENT.md)

**v3.1 (2026-10-25): the reference GPU is the H200 SXM (`NVIDIA H200`). The H100 records keep a
separate board. v3.2 (2026-10-26): an untimed warm-up is allowed (RULES_v3, "Untimed warm-up"); record
seeds now run `scripts/warm_start.sh` first and log its time to `<record>/warm_start.log`.**

| record | config | runs on | state |
|---|---|---|---|
| `records/v3/track_a/004_ladder_constrained_h200` | A 001's, unchanged | any H200 worker (front of `h200`) | seed 0 preempted at 99% (10-28 00:25), reruns first on 26824150; seeds 1-4 queued |
| `records/v3/track_d/004_dm_math_fp32_h200_warm` | D 002's, unchanged | preemptable H200 workers 26933254 (+ 27162158 after the probe) | seed 1 done: test 0.1296, within budget; seed 0 reruns on 26933254; seeds 2-4 queued |
| `records/v3/track_a/002_fp32_skip_cosine` | as promoted 10-23 | any H200 worker | queued at the end of `h200` (LOG 13:05) |
| `records/v3/track_d/003_dm_math_fp32_h200` | D 002's, unchanged | done | **failed**: seeds 0 and 2 over budget; README reports it (test mean of all five 0.1538) |

- A preempted seed is an infrastructure stop: its partial attempt goes to `extra/` and the same seed
  reruns (INBOX 10-25 11:15).
- Promotion uses the H100 lines until the reruns land: A 0.2703 (3-seed form: every seed at least
  that), D 0.190. 5-seed form: mean at least the line, at least 4 of 5 seeds above the holder's dev
  mean (A 0.240, D 0.160), at most one collapsed. When A 004 and D 004 land, their in-loop final dev
  replaces these (PLAN).
- node4900 is excluded from every worker (CPUs oversubscribed, LOG 10-25 14:30).

H100 board (closed 10-24): A 001 holds track A (test 0.2481, dev 0.2403); D 002 holds track D
(test 0.1512 ± 0.0202, dev 0.160).

Promoted on H100 dev numbers, not yet recorded: A 002 (dev 0.324) and A 003 (dev 0.324).

Updated 2026-10-28 19:35 EDT.

## AWS boxes: off

INBOX 10-25 22:00: the capacity blocks are under a refund review and must stay unused. No box
launches and no AWS spending, including the $500 emergency budget, until a later INBOX note says
otherwise. `queue/b200.txt` (r15) is parked.

## Running

- mit_preemptable H200 (`--time-min=06:30:00`, RESERVE_MIN 380, NVL nodes and node4900 excluded):
  - Pending after the 10-27/28 preemptions (LOG 10-28 19:25): 26824150 (A 004 seed 0 rerun first),
    26854533 (r16 seed 4 from step ~296), 26933254 (D 004 seed 0 rerun first).
  - 27162158: waits on the probe; D 004's queue, then `h200`.
- mit_normal_gpu H200 (6 h, one item per job): two chains of three, heads 26927304 and 26927308
  (pending); they read A 004's and A 002's (empty) record queues, then `h200`.
- B200 packing probe 27162075 (mit_preemptable, 2.5 h, pending): `pack_probe.sh`, now with GCC 12's
  libstdc++ first (26953974 failed on it; a CPU check confirmed the fix). 1 then 2 track A runs and 1 then 4 track D
  runs; it prints steps/h (LOG 10-26 17:45, 10-28 19:25).
- Cap: 0 running + 6 pending (3 preemptable H200 workers, 2 normal heads, the probe); 27162158
  waits on the probe. I may
  update or cancel my own pending jobs without asking.

## Queues

- `h200` (25): A 004 seeds 1-4; r17 `mal0_st500_trs05` / `trs02` seeds 0-2; r18 `d003rl_p256_lr3e-6`
  / `lr6e-6` seeds 1-3; r17 seeds 3-4; A 002 seeds 0-4. Every H200 worker reads it.
- `record-004_dm_math_fp32_h200_warm` (3): D 004 seeds 2-4 (read by 26933254 and 27162158 only).
- Other record queues: empty. `b200` (r15), `h100`, `later`: parked, read by no worker.

## Best so far

- Track A (dev): r16 (r5 + cosine over 500 steps) 0.232 / 0.405 / 0.412 / 0.332, seed 4 pending;
  r14 (r5 + TIS, same lr path by accident) 0.398 / 0.403. A 002 and A 003 0.324 (H100). r16 seed 0
  slid into truncation in its last 50 steps (truncated 0.54 to 0.64), which r17 targets.
- Track D: D 002's recipe, test 0.1512 (H100 record) and 0.1538 (D 003 on the H200, invalid clocks).
  Closed: dm51, r11 (lr 1e-5, 3e-5).

## Next

- When 26854533 restarts: r16 seed 4 ends ~90 min later. r16 promotes if it ends above 0.240 without collapsing.
  Then its record (r16's config plus `budget_steps_time_stop`) goes into `h200` right after A 004.
- Check each record seed's `clock.json` `within_budget` and `warm_start.log` as soon as it ends.
- When the B200 probe reports: compare its steps/h with r16's H200 log at matched output lengths
  (LOG 10-26 17:45). If two runs per B200 pay off, step-capped track A screens move to screen-only B200
  workers (two per GPU, r15's memory settings) from a queue without record items; time-budgeted
  screens (r18) stay on the H200.
- r17 seed 0s start once A 004's seeds 1-4 are all taken from `h200`.

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q` (two-rank tests fail on the login node otherwise).
- Eval arms: a dir with `eval.json`, queued like a screen, runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`). `summarize_screens.py` prints them.
- Step-capped runs (`budget_steps`) log `clock_s` as the step clock (500 steps = 17,400 s);
  `summarize_screens.py` reads their minutes from it. `budget_steps_time_stop` also stops them at
  `time_budget_s` of real time (for records).
- Track D pretraining is time-budgeted and its speed varies by node; read D results with their
  tokens/s.
- Ubuntu 24 nodes: the worker adds gcc's multiarch paths; `module load` fails there.
- Record seeds: check `clock.json` `within_budget` as soon as a seed finishes.
- Workers exit on empty queues: keep `h200` stocked ahead of every worker's next pop.
