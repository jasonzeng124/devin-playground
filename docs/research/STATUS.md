# Status

(maintained by the research agent; see AGENT.md)

**v3.1 (2026-10-25): the reference GPU is the H200 SXM (`NVIDIA H200`). The H100 records keep a
separate board. v3.2 (2026-10-26): an untimed warm-up is allowed (RULES_v3, "Untimed warm-up"); record
seeds now run `scripts/warm_start.sh` first and log its time to `<record>/warm_start.log`.**

| record | config | runs on | state |
|---|---|---|---|
| `records/v3/track_a/004_ladder_constrained_h200` | A 001's, unchanged | any H200 worker (front of `h200`) | seed 0's rerun runs on 26824150 (node4301, from 10-28 21:36); its preempted attempt is in `extra/interrupted_0_1`; seed 1 runs on 26854533 (node4301, from 23:06); seeds 2-4 queued |
| `records/v3/track_a/005_fp32_mal0_cos500` | r16's + `budget_steps_time_stop`, weights saved | any H200 worker | **new** (LOG 10-28 23:10): r16 promoted; seeds 0-4 queued in `h200` after A 004 |
| `records/v3/track_d/004_dm_math_fp32_h200_warm` | D 002's, unchanged | preemptable H200 workers 26933254 (+ 27162158 after the B200 worker) | seed 1 done: test 0.1296, within budget; seed 0 reruns on 26933254; seeds 2-4 queued |
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

Promoted, not yet recorded: A 002 and A 003 (H100 dev 0.324 each); A 005 (r16, step-capped dev
0.364, 10-28).

Updated 2026-10-28 23:15 EDT.

## AWS boxes: off

INBOX 10-25 22:00: the capacity blocks are under a refund review and must stay unused. No box
launches and no AWS spending, including the $500 emergency budget, until a later INBOX note says
otherwise. `queue/b200.txt` (r15) is parked.

## Running

- mit_preemptable H200 (`--time-min=06:30:00`, RESERVE_MIN 380, NVL nodes and node4900 excluded):
  - 26854533: A 004 seed 1 (node4301, from 23:06).
  - 26824150: A 004 seed 0's rerun (node4301, from 21:36).
  - 26933254 (pending): D 004 seed 0's rerun first.
  - 27162158: waits on B200 worker 27168549 (afterany); D 004's queue, then `h200`.
- mit_normal_gpu H200 (6 h, one item per job): two chains of three, heads 26927304 (est. 10-29 18:00)
  and 26927308 (waits on B200 worker 27168550); they read A 004's and A 002's (empty) record queues,
  then `h200`.
- mit_preemptable B200 screen workers 27168549 and 27168550 (LOG 10-28 20:25): `packed_worker.sbatch`
  with PER_GPU=1 (GCC 12's libstdc++ first, deadline from `squeue`), REQUIRE_GPU "NVIDIA B200",
  RESERVE_MIN 240, queue `screens_b200` only. One run per GPU: the probe found a second track A run
  adds 11 %.
- Cap: 4 running (2 B200, 2 H200) + 2 eligible pending (26933254, 26927304); 27162158 and 26927308 wait
  on the B200 workers. I may update or cancel my own pending jobs without asking.

## Queues

- `h200` (19): A 004 seeds 2-4; A 005 seeds 0-4; r18 `d003rl_p256_lr3e-6` / `lr6e-6` seeds 1-3;
  A 002 seeds 0-4. Every H200 worker reads it.
- `screens_b200` (7): r17 `mal0_st500_trs05` / `trs02`, read by the B200 workers only (no record items,
  no time-budgeted screens). Done: `trs05` seed 1, dev 0.357. Running: `trs02` seed 0 (node5501-c1,
  dev 0.395 at step 400), `trs05` seed 0 (node5600-c1, from 22:29). Queued: `trs02` seed 1, then
  seeds 2-4 of both.
- `record-004_dm_math_fp32_h200_warm` (3): D 004 seeds 2-4 (read by 26933254 and 27162158 only).
- Other record queues: empty. `b200` (r15), `h100`, `later`: parked, read by no worker.

## Best so far

- Track A (dev): r16 (r5 + cosine over 500 steps) 0.232 / 0.405 / 0.412 / 0.332 / 0.438, mean 0.364:
  promoted to A 005. r14 (r5 + TIS, same lr path by accident) 0.398 / 0.403. A 002 and A 003 0.324
  (H100). r16's seeds that kept about a fifth of outputs truncated ended highest; seed 0 slid into
  truncation in its last 50 steps (0.54 to 0.61), which r17 targets.
- Track D: D 002's recipe, test 0.1512 (H100 record) and 0.1538 (D 003 on the H200, invalid clocks).
  Closed: dm51, r11 (lr 1e-5, 3e-5).

## Next

- A 005's seeds follow A 004's in `h200`. Check each record seed's `clock.json` `within_budget` and
  `warm_start.log` as soon as it ends; an A 005 seed should end about 200 min into its 300-min budget.
- r17 (B200, step-capped): its line is effectively r16's dev, since A 005 becomes the holder if it
  lands (LOG 23:10). If an arm clears r16 clearly, three seeds rerun on the H200 before any record.
- Track D packs well on a B200 (three runs 2.5x one run's steps/h), but only for step-capped RL
  screens; r18 is time-budgeted and stays on the H200.

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
- B200 workers: start them a few minutes apart (simultaneous first Triton compiles race on the shared
  cache, LOG 10-28 20:34).
