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
| `records/v3/track_a/005_fp32_mal0_cos500_s400` | r16's, stopped at step 400 (`max_steps` 400), + `budget_steps_time_stop` | done | **Complete: 0.3400 ± 0.0815, validated, not a new record** (Welch vs A 004 p = 0.072; LOG 11-01 09:50). Seeds 0.3995 / 0.4076 / 0.3821 / 0.2209 / 0.2899; dev mean 0.313; seeds 3-4 shortened their outputs; four preempted attempts in `extra/` |
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

Promoted, not yet recorded: A 002 and A 003 (H100 dev 0.324 each).

Updated 2026-11-01 17:45 EST (r26 H200 seed 2 ended at dev 0.148 / 0.183 for seeds 1 / 2 against D 004's 0.130 / 0.137, still below the 0.189 line; r27 `shape05_lr6e-6` complete at step 6,000, mean 0.169; r30/r31 full-length lr 6e-6 runs queued).

## AWS boxes: off

INBOX 10-25 22:00: the capacity blocks are under a refund review and must stay unused. No box
launches and no AWS spending, including the $500 emergency budget, until a later INBOX note says
otherwise. `queue/b200.txt` (r15) is parked.

## Running

- mit_preemptable H200 (`--time-min=06:30:00`, RESERVE_MIN 380, NVL nodes and node4900 excluded):
  - 26854533 and 26824150 exited 10-30 22:20 after A 004 seeds 3 and 2 (less than 380 min left).
  - 27412796 (submitted 10-30 22:50, excludes node2500 and node4900): ran A 005 seed 3's rerun on node4801 (11-01 06:46-09:35, test 0.2209); now r26 seed 0.
  - 27412797 (same): ran A 005 seed 2's rerun (test 0.3821); exited 11-01 03:40 EST (less than 380 min left).
  - 27530015 (submitted 11-01 03:45 EST, same settings, queue `h200`): pending.
  - 26933254 exited 10-29 20:29 after D 004 seed 2.
  - 27162158 (preempted 10-30 21:03 and 10-31 09:03): ran A 005 seed 4 (test 0.2899) and r18 `d003rl_p256_lr3e-6` seed 1 (dev 0.148); exited 10-31 19:07 (less than 380 min left).
  - 27487788 (submitted 10-31 19:50, same settings, queue `h200`): running on node4300 since about 09:00 EST 11-01; r26 seeds 1 and 2 done, now r18 `lr6e-6` seed 1.
- mit_normal_gpu H200 (6 h, one item per job): two chains. 26927304 ran A 004 seed 4 (10-29 08:07-13:07);
  26927305 is next (eligible), then 26927306. The other chain (26927308 -> 26927310 -> 26927311) was cancelled 10-31 12:48 to free a slot for track D. They read A 004's and A 002's (empty) record queues, then `h200` (head: A 005 seed 0).
- mit_preemptable B200 screen worker 27318631 (exited 10-30 16:58 with less than 240 min left; from 10-30 02:48, replaces 27168550, which was requeued onto node5501-c1 and exited there 10-30 02:10, LOG 02:50; 27168549 exited 10-29 04:43 on node5501-c1's failed GPU;
  27206873 cancelled 10-29 12:05 after a preemption, LOG 12:10) (LOG 10-28 20:25): `packed_worker.sbatch`
  with PER_GPU=1 (GCC 12's libstdc++ first, deadline from `squeue`), REQUIRE_GPU "NVIDIA B200",
  RESERVE_MIN 240, queue `screens_b200` only. One run per GPU: the probe found a second track A run
  adds 11 %.
- Second B200 screen worker 27283068 (same settings, from 10-29 20:32; preempted and requeued 10-30 00:35, 01:38, about 02:00 and about 20:30; exited 10-31 03:36 after r24 seed 0, less than 240 min left). 27432002 (10-31 03:54-06:10) ran r24 seed 1 and exited the same way. 27441594 (10-31 06:48 to 11-01 05:45) kept resuming r22 seed 3 from its only save (step 224) and was cancelled. 27534692 (submitted 11-01 05:45 EST) uses the same settings plus `RESUME_EVERY_S=1800`.
- Track D B200 worker 27483976 (submitted 10-31 18:50, replaces 27460856, whose eight preemptions on node5800-c1 each came before the first hourly save, LOG 18:50): `packed_worker.sbatch` with PER_GPU=3, RESERVE_MIN 180, `--time-min=08:00:00`, `RESUME_EVERY_S=600`, queue `screens_d_b200` (r25); running again since 11-01 00:17, resumed from the saves.
- Cap: 3 running (27487788, 27483976, 27585540, which replaced 27534692 at 16:45 EST) + 3 pending that could run together (27412796, 26927305, 27530015). 27412796 was preempted about 10:30 EST and holds r26 seed 0 (resumes from its save). Both B200 workers on node5800-c1 were preempted and requeued four times 11:00-11:30 EST.
  I may update or cancel my own pending jobs without asking.

## Queues

- `h200` (9 + 2 running): r26 `d004rl_shape05_h200` seed 0 (preempted, held by 27412796; D 004's full RL with shaping 0.5; seeds 1 / 2 ended 0.148 / 0.183, so it cannot promote, and seed 0 runs as paired evidence); r18 `lr3e-6` seed 2 and `lr6e-6` seed 2 running, then `lr3e-6` seed 3, `lr6e-6` seed 3 (seed 1: `lr3e-6` 0.148, `lr6e-6` 0.142, D 003's pair 0.125; LOG 11-01 21:45);
  A 002 seeds 0-4. Every H200 worker reads it.
- `screens_d_b200` (5 queued, none running; worker 27598456 pending): 27483976 ended 11-01 19:39 EST; its successor 27598456 (same settings) is pending; then r31 `d004rl_shape05_lr6e-6_cos_s15k` seeds 0-2 interleaved with r30 `d004rl_shape05_lr6e-6_s15k` seeds 0-1 (15,000 steps, about 6.3 h each; LOG 11-01 17:45).
  At step 6,000: r25 `shape05` 0.188 / 0.140 / 0.153, `shape02` 0.140 / 0.185 / 0.148; r27 `shape05_lr6e-6` 0.175 / 0.152 / 0.180; r28 `shape05_g16` 0.138 / 0.158 / 0.168, closed (LOG 11-01 18:45); r29 `shape05_lr1e-5` 0.170 / 0.172 / 0.182 (LOG 11-01 19:45); D 004's own dev 0.122 / 0.115 / 0.075 (seeds 0 / 1 / 2). `shape10` 0.123 / 0.118, closed.
- `screens_b200` (1 + 1 running, worker 27585540): r30 seed 2 running (step-capped, so it compares with the packed seeds by step). r22 `ent05_t10_lr2e-6` closed: 0.360 / 0.387 / 0.440 / 0.357 / 0.470, mean 0.403 (LOG 11-01 11:50). r24 `ent05_lr3e-6` closed: 0.413, 0.065
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
