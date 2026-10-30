# Status

(maintained by the research agent; see AGENT.md)

**v3.1 (2026-10-25): the reference GPU is the H200 SXM (`NVIDIA H200`). The H100 records keep a
separate board. v3.2 (2026-10-26): an untimed warm-up is allowed (RULES_v3, "Untimed warm-up"); record
seeds now run `scripts/warm_start.sh` first and log its time to `<record>/warm_start.log`.**

| record | config | runs on | state |
|---|---|---|---|
| `records/v3/track_a/004_ladder_constrained_h200` | A 001's, unchanged | any H200 worker (front of `h200`) | seed 0 done: test 0.3135, dev 0.287, within budget (17,463 s); its preempted attempt is in `extra/interrupted_0_1`; seed 1 done: test 0.2358, dev 0.230, within budget; seeds 2 and 3 were preempted on 10-29 04:48 (their workers, 26824150 and 26854533, rerun them on restart and keep the partial runs under `extra/`); seeds 2 and 3 reran on 26824150 and 26854533 from 10-30 01:59 and 04:30 and were both preempted again at 04:53 (attempts 2 at steps 728 and 75; attempts 1 in `extra/interrupted_{2,3}_1`), and wait for those workers to restart; seed 4 done: test 0.1834 (one-guess), dev 0.185, within budget |
| `records/v3/track_a/005_fp32_mal0_cos500_s400` | r16's, stopped at step 400 (`max_steps` 400), + `budget_steps_time_stop`, weights saved | any H200 worker | **new** (LOG 10-28 23:10, 10-29 00:30): r16 promoted; r16's dev at step 400 0.397 ± 0.011; seeds 0-4 queued in `h200` after A 004 |
| `records/v3/track_d/004_dm_math_fp32_h200_warm` | D 002's, unchanged | preemptable H200 worker 27162158 | seed 1 done: test 0.1296, within budget; seed 0 done (third attempt; attempt 2 in `extra/interrupted_0_2`): test 0.1832, dev 0.182, within budget (20,816 s); seed 2 done: test 0.1384, dev 0.137, within budget (20,892 s); seed 3 running on 27162158 (node2500, from 10-30 06:42 after a 59-s warm start); seed 4 queued |
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

Promoted, not yet recorded: A 002 and A 003 (H100 dev 0.324 each); A 005 (r16 stopped at step 400,
step-capped dev 0.397, 10-29).

Updated 2026-10-30 07:50 EDT (D 004 seed 3 running; r22 `ent01` seed 1 collapsed; r23 is now grad_clip 0.3; train_log has `grad_norm`).

## AWS boxes: off

INBOX 10-25 22:00: the capacity blocks are under a refund review and must stay unused. No box
launches and no AWS spending, including the $500 emergency budget, until a later INBOX note says
otherwise. `queue/b200.txt` (r15) is parked.

## Running

- mit_preemptable H200 (`--time-min=06:30:00`, RESERVE_MIN 380, NVL nodes and node4900 excluded):
  - 26854533 (pending, preempted 10-30 04:53): reruns A 004 seed 3 first.
  - 26824150 (pending, preempted 10-30 04:53): reruns A 004 seed 2 first.
  - 26933254 exited 10-29 20:29 after D 004 seed 2.
  - 27162158 (running on node2500 from 10-30 06:42): D 004 seed 3, then seed 4, then `h200`.
- mit_normal_gpu H200 (6 h, one item per job): two chains. 26927304 ran A 004 seed 4 (10-29 08:07-13:07);
  26927305 is next (eligible), then 26927306. The other chain, 26927308 -> 26927310 -> 26927311, waits on B200
  worker 27318631. They read A 004's and A 002's (empty) record queues, then `h200` (head: A 005 seed 0).
- mit_preemptable B200 screen worker 27318631 (from 10-30 02:48, replaces 27168550, which was requeued onto node5501-c1 and exited there 10-30 02:10, LOG 02:50; 27168549 exited 10-29 04:43 on node5501-c1's failed GPU;
  27206873 cancelled 10-29 12:05 after a preemption, LOG 12:10) (LOG 10-28 20:25): `packed_worker.sbatch`
  with PER_GPU=1 (GCC 12's libstdc++ first, deadline from `squeue`), REQUIRE_GPU "NVIDIA B200",
  RESERVE_MIN 240, queue `screens_b200` only. One run per GPU: the probe found a second track A run
  adds 11 %.
- Second B200 screen worker 27283068 (same settings, from 10-29 20:32; preempted and requeued 10-30 00:35, 01:38 and about 02:00, now on node5800-c1).
- Cap: 3 running (27162158, 27283068, 27318631) + 3 eligible pending (26824150, 26854533, 26927305);
  26927308 waits on B200 worker 27318631, which exits when `screens_b200` is empty. I may update or cancel my own pending jobs without asking.

## Queues

- `h200` (16): A 005 (`005_fp32_mal0_cos500_s400`) seeds 0-4; r18 `d003rl_p256_lr3e-6` / `lr6e-6` seeds 1-3;
  A 002 seeds 0-4. Every H200 worker reads it.
- `screens_b200` (5 + 2 running): r22 `ent01_t10_lr2e-6` seeds 3-4 (seed 2 running) and `ent05_t10_lr2e-6` seed 2 (running);
  r23 `ent01_t10_lr2e-6_gc03` (grad_clip 0.3) seeds 0-2 (LOG 10-30 07:50). r22 so far: `ent01` s0 0.427 (0.458 at step 400),
  s1 0.078 (truncation-led collapse at step 350); `ent05` 0.360, 0.387; all peak at steps 300-400. r21 closed: `ent01_t10` 0.369 ± 0.058, `ent05_t10` 0.399 ± 0.011 (LOG 10-30 01:45). r20, r17, r19 closed.
- `record-004_dm_math_fp32_h200_warm` (1): D 004 seed 4 (seed 3 running; read by 27162158 only).
- Other record queues: empty. `b200` (r15), `h100`, `later`: parked, read by no worker.

## Best so far

- Track A (dev): r16 (r5 + cosine over 500 steps) 0.232 / 0.405 / 0.412 / 0.332 / 0.438, mean 0.364;
  at step 400 0.397 ± 0.011, which A 005 uses (stop at 400). r14 (r5 + TIS, same lr path by accident) 0.398 / 0.403. A 002 and A 003 0.324
  (H100). r16's seeds that kept about a fifth of outputs truncated ended highest; seed 0 slid into
  truncation in its last 50 steps (0.54 to 0.61), which r17 targets.
- Track D: D 002's recipe, test 0.1512 (H100 record) and 0.1538 (D 003 on the H200, invalid clocks).
  Closed: dm51, r11 (lr 1e-5, 3e-5).

## Next

- A 005's seeds follow A 004's in `h200`. Check each record seed's `clock.json` `within_budget` and
  `warm_start.log` as soon as it ends; an A 005 seed should train about 2.7 h (400 steps).
- Every slide so far (r16, r17, r19, r20 `p96`) starts within 50-100 steps of entropy reaching
  0.035-0.045. Next: r21, an entropy bonus that turns on below a target entropy (new option, off by
  default), run on the B200 workers (r21 closed, r22 running). A new arm needs about 0.43 to beat A 005's expected 0.397; its
  gating seeds then run on the H200.
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
