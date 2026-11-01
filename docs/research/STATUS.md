# Status

(maintained by the research agent; see AGENT.md)

**v3.3 (2026-10-31): external data is withdrawn for track A (generator puzzles and verifier verdicts only) and kept for track D.**

**INBOX 10-31 12:15: track D matters more; it gets at least half of research effort and screen capacity. A 005 and A 002 still run.**

**v3.1 (2026-10-25): the reference GPU is the H200 SXM (`NVIDIA H200`). The H100 records keep a
separate board. v3.2 (2026-10-26): an untimed warm-up is allowed (RULES_v3, "Untimed warm-up"); record
seeds now run `scripts/warm_start.sh` first and log its time to `<record>/warm_start.log`.**

| record | config | runs on | state |
|---|---|---|---|
| `records/v3/track_a/004_ladder_constrained_h200` | A 001's, unchanged | done | **Complete: 0.2671 ± 0.0565, validated, track A (H200) holder (LOG 10-30 22:55)**. Seeds 0.3135 / 0.2358 / 0.3132 / 0.2895 / 0.1834 (seed 4 one-guess collapse); dev mean 0.255; five preempted attempts in `extra/` |
| `records/v3/track_a/005_fp32_mal0_cos500_s400` | r16's, stopped at step 400 (`max_steps` 400), + `budget_steps_time_stop`, weights saved | any H200 worker | **new** (LOG 10-28 23:10, 10-29 00:30): r16 promoted; r16's dev at step 400 0.397 ± 0.011; seed 0's attempts 1 (node2500, step 165) and 2 (step 1) were preempted and are in `extra/`. **Seed 0: test 0.3995, seed 1: 0.4076**, both within budget (LOG 10-31 12:50). Seed 2 trained (dev 0.36) but was preempted in its test ladder: it reruns when 27412797 restarts. Seed 3 preempted at step 130 (reruns when 27412796 restarts). **Seed 4: test 0.2899**, within budget (LOG 10-31 15:50) |
| `records/v3/track_d/004_dm_math_fp32_h200_warm` | D 002's, unchanged | done | seed 1 done: test 0.1296, within budget; seed 0 done (third attempt; attempt 2 in `extra/interrupted_0_2`): test 0.1832, dev 0.182, within budget (20,816 s); seed 2 done: test 0.1384, dev 0.137, within budget (20,892 s); seed 3 done: test 0.1677, dev 0.183, within budget (20,941 s); seed 4 done: test 0.1572, dev 0.163, within budget (20,932 s). **Complete: 0.1552 ± 0.0217, validated, track D (H200) holder (LOG 10-30 18:55)** |
| `records/v3/track_a/002_fp32_skip_cosine` | as promoted 10-23 | any H200 worker | queued at the end of `h200` (LOG 13:05) |
| `records/v3/track_d/003_dm_math_fp32_h200` | D 002's, unchanged | done | **failed**: seeds 0 and 2 over budget; README reports it (test mean of all five 0.1538) |

- A preempted seed is an infrastructure stop: its partial attempt goes to `extra/` and the same seed
  reruns (INBOX 10-25 11:15).
- Promotion lines (H200 holders' dev mean + 0.03, 10-30): A 0.285 (A 004; 3-seed form: every seed at
  least that), D 0.189 (D 004). 5-seed form: mean at least the line, at least 4 of 5 seeds above the holder's dev
  mean (A 0.255, D 0.159), at most one collapsed. When A 004 and D 004 land, their in-loop final dev
  replaces these (PLAN).
- node4900 is excluded from every worker (CPUs oversubscribed, LOG 10-25 14:30). node2500 is excluded from
  every H200 worker (track A ran about 3x slow there, LOG 10-30 21:55).

H200 board: A 004 holds track A (test 0.2671 ± 0.0565, dev 0.255); D 004 holds track D (test
0.1552 ± 0.0217, dev 0.159).

H100 board (closed 10-24): A 001 holds track A (test 0.2481, dev 0.2403); D 002 holds track D
(test 0.1512 ± 0.0202, dev 0.160).

Promoted, not yet recorded: A 002 and A 003 (H100 dev 0.324 each); A 005 (r16 stopped at step 400,
step-capped dev 0.397, 10-29).

Updated 2026-11-01 00:45 EDT (B200 workers restarted 00:17; r25 resumed from its 10-min saves, `shape05` dev 0.137 at step 1,000 against D 004's 0.087; A 005 seeds 2-3 and the H200 worker still pending).

## AWS boxes: off

INBOX 10-25 22:00: the capacity blocks are under a refund review and must stay unused. No box
launches and no AWS spending, including the $500 emergency budget, until a later INBOX note says
otherwise. `queue/b200.txt` (r15) is parked.

## Running

- mit_preemptable H200 (`--time-min=06:30:00`, RESERVE_MIN 380, NVL nodes and node4900 excluded):
  - 26854533 and 26824150 exited 10-30 22:20 after A 004 seeds 3 and 2 (less than 380 min left).
  - 27412796 (submitted 10-30 22:50, excludes node2500 and node4900): preempted 12:48 with A 005 seed 3 at step 130; pending, holds seed 3.
  - 27412797 (same): preempted 12:22 in A 005 seed 2's test ladder; pending, holds seed 2.
  - 26933254 exited 10-29 20:29 after D 004 seed 2.
  - 27162158 (preempted 10-30 21:03 and 10-31 09:03): ran A 005 seed 4 (test 0.2899) and r18 `d003rl_p256_lr3e-6` seed 1 (dev 0.148); exited 10-31 19:07 (less than 380 min left).
  - 27487788 (submitted 10-31 19:50, same settings, queue `h200`): pending.
- mit_normal_gpu H200 (6 h, one item per job): two chains. 26927304 ran A 004 seed 4 (10-29 08:07-13:07);
  26927305 is next (eligible), then 26927306. The other chain (26927308 -> 26927310 -> 26927311) was cancelled 10-31 12:48 to free a slot for track D. They read A 004's and A 002's (empty) record queues, then `h200` (head: A 005 seed 0).
- mit_preemptable B200 screen worker 27318631 (exited 10-30 16:58 with less than 240 min left; from 10-30 02:48, replaces 27168550, which was requeued onto node5501-c1 and exited there 10-30 02:10, LOG 02:50; 27168549 exited 10-29 04:43 on node5501-c1's failed GPU;
  27206873 cancelled 10-29 12:05 after a preemption, LOG 12:10) (LOG 10-28 20:25): `packed_worker.sbatch`
  with PER_GPU=1 (GCC 12's libstdc++ first, deadline from `squeue`), REQUIRE_GPU "NVIDIA B200",
  RESERVE_MIN 240, queue `screens_b200` only. One run per GPU: the probe found a second track A run
  adds 11 %.
- Second B200 screen worker 27283068 (same settings, from 10-29 20:32; preempted and requeued 10-30 00:35, 01:38, about 02:00 and about 20:30; exited 10-31 03:36 after r24 seed 0, less than 240 min left). 27432002 (10-31 03:54-06:10) ran r24 seed 1 and exited the same way. 27441594 (submitted 10-31 06:48, `--time-min=08:00:00` so it runs two screens per start) replaces it; preempted 7 times on node5800-c1 since 06:49, running r22 `ent05` seed 3 again from 10:36; pending again at 12:43.
- Track D B200 worker 27483976 (submitted 10-31 18:50, replaces 27460856, whose eight preemptions on node5800-c1 each came before the first hourly save, LOG 18:50): `packed_worker.sbatch` with PER_GPU=3, RESERVE_MIN 180, `--time-min=08:00:00`, `RESUME_EVERY_S=600`, queue `screens_d_b200` (r25); running again since 11-01 00:17, resumed from the saves.
- Cap: 2 running (27441594, 27483976, both on node5800-c1 since 11-01 00:17) + 4 pending that could run together (27412796, 27412797, 26927305, 27487788).
  I may update or cancel my own pending jobs without asking.

## Queues

- `h200` (10): r18 `lr6e-6` seeds 1-3, `lr3e-6` seeds 2-3 (`lr3e-6` seed 1: dev 0.148, D 003's pair 0.125, LOG 19:50);
  A 002 seeds 0-4. Every H200 worker reads it.
- `screens_d_b200` (10, worker 27483976): r25, track D RL from D 004's pretrained seeds 0-2, 6,000 steps:
  `d004rl_shape02`, `d004rl_shape05`, `d004rl_ent05_t10` seeds 0-2, `d004rl_ctl` seed 0. Paired control: D 004's
  own dev at step 6,000 (0.122 / 0.115 / 0.075, LOG 10-31 12:50).
- `screens_b200` (1 + 1 held, one worker): r22 `ent05_t10_lr2e-6` seed 4 (seed 3 running). r24 `ent05_lr3e-6` closed: 0.413, 0.065
  (collapse at step 430; LOG 10-31 06:50). r23 closed: `gc015` 0.392, 0.018, 0.005, 0.435 (LOG 10-30 17:55);
  `gc03` 0.378, 0.125 (LOG 10-31 01:50).
  r22 closed: `ent01` 0.355 ± 0.157, `ent05` 0.396 ± 0.041. ent01-based runs: 3 collapses in 10. r21, r20, r17, r19 closed.
- Other record queues: empty. `b200` (r15), `h100`, `later`: parked, read by no worker.

## Best so far

- Track A (dev): r16 (r5 + cosine over 500 steps) 0.232 / 0.405 / 0.412 / 0.332 / 0.438, mean 0.364;
  at step 400 0.397 ± 0.011, which A 005 uses (stop at 400). r14 (r5 + TIS, same lr path by accident) 0.398 / 0.403. A 002 and A 003 0.324
  (H100). r16's seeds that kept about a fifth of outputs truncated ended highest; seed 0 slid into
  truncation in its last 50 steps (0.54 to 0.61), which r17 targets.
- Track D: D 002's recipe, test 0.1552 ± 0.0217 (D 004, the H200 record), 0.1512 (H100 record) and
  0.1538 (D 003 on the H200, invalid clocks).
  Closed: dm51, r11 (lr 1e-5, 3e-5).

## Next

- A 005's seeds follow A 004's in `h200`. Check each record seed's `clock.json` `within_budget` and
  `warm_start.log` as soon as it ends; an A 005 seed should train about 2.7 h (400 steps).
- Every slide so far (r16, r17, r19, r20 `p96`) starts within 50-100 steps of entropy reaching
  0.035-0.045. Next: r21, an entropy bonus that turns on below a target entropy (new option, off by
  default), run on the B200 workers (r21 closed, r22 running). A new arm needs about 0.43 to beat A 005's expected 0.397; its
  gating seeds then run on the H200.
- Track D: every D 004 seed ends one-guess, with 1-10 informative groups of 64 from step ~1,500 on. r25 tests
  partial credit (`shaping`) and the entropy bonus against that. Next directions: PLAN D3 (SFT on a published
  worked-solutions set) and D4 (process rewards). r18 is time-budgeted and stays on the H200.

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
