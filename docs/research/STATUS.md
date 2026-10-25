# Status

(maintained by the research agent; see AGENT.md)

**v3.1 (2026-10-25): the reference GPU is the H200 SXM (`NVIDIA H200`). The H100 records keep a
separate board. The H200 board starts with three record runs, all queued (LOG 01:24):**

| record | config | runs on | first start (estimate) |
|---|---|---|---|
| `records/v3/track_a/004_ladder_constrained_h200` | A 001's, unchanged | preemptable H200 workers and mit_normal_gpu chain 1 (26824033-37) | seed 0 ~21:25 tonight (26824150) |
| `records/v3/track_a/002_fp32_skip_cosine` | as promoted 10-23 | preemptable H200 workers and mit_normal_gpu chain 2 (26824038-42) | seed 0 ~02:45 on 10-26 |
| `records/v3/track_d/003_dm_math_fp32_h200` | D 002's, unchanged | mit_preemptable H200 (26824148, 26824150) | seeds 0-1 running since 08:09 / 08:43 |

- Since 11:46, track A's H200 record seeds also run on mit_preemptable H200 workers (INBOX 11:15,
  LOG 11:46). A preempted seed is an infrastructure stop: its partial attempt goes to `extra/` and
  the same seed reruns.
- A 004 and A 002 run side by side. Chain 1 serves A 004 first and chain 2 serves A 002 first; each
  helps the other when its own queue is empty. A 002 is compared with A 004 once both land.
- A 003 (`records/v3/track_a/003_fp32_mal0_s300`) waits, as the laptop asked.
- Promotion uses the H100 lines until the reruns land: A 0.2703 (163 of 600 per seed for the
  3-seed form), D 0.190 (115 of 600). Then the lines come from the reruns' in-loop final dev
  (PLAN).

H100 board (closed 10-24):
- A 001 is the holder (test 0.2481, dev 0.2403).
- D 002 landed 10-24 22:39 (test 0.1512 ± 0.0202, p = 0.0092 against D 000; dev 0.160).

Promoted on H100 dev numbers, not yet recorded:
- A 002 (r2 `a01_fp32_lr1e-6_skip_cos`): dev 0.324, seeds 0.300 / 0.270 / 0.472 / 0.315 / 0.263.
  It now runs on the H200 (above).
- A 003 (r7 `a01_fp32_lr1e-6_skip_cos_mal0_s300`): dev 0.324, seeds 0.385 / 0.348 / 0.238 /
  0.348 / 0.302. It stops at step 300; about one seed in five of its recipe fell between steps
  250 and 300 (LOG 10-24 00:55).

Updated 2026-10-25 11:50 EDT.

## Running

- mit_preemptable H100 workers (`main:long`):
  - 26786843 (node2700): r13 seed 1 (r10 + KL 0.05) since 08:06, ends ~13:15. Before it: r11
    lr 1e-5 seed 0 (0.142) and c90_pos01 seed 0 (one-guess).
  - 26803673 (node2905): dm51 seed 2 since 09:06, ends ~13:20. Before it: r11 lr 3e-5 seed 0
    (0.068).
  - At ~13:20 they take r10 seed 2 and r14 seed 0 (r5 + truncated importance sampling) from `main`
    instead of exiting; r14 seed 1 follows at ~18:30. Their H200 successors would wait anyway
    (Slurm's estimate for a new H200 job at 09:15: 10-26 00:51).
  - 26786872 exited at 05:00 after r10 seed 1, and 26786844 (r9) was cancelled at 02:27.
- H200 workers on mit_preemptable (`--time-min=06:30:00`, RESERVE_MIN 380, NVL nodes excluded):
  - 26824148 runs D 003 seed 0 on node4900 since 08:09 (at 11:45: 90 RL min, dev 0.120), and
    26824150 runs seed 1 on node5001 since 08:43 (64 RL min, dev 0.130). Both read
    `record-003_dm_math_fp32_h200:h200`.
  - They take seeds 2-3 at ~14:30-15:05 and seed 4 at ~20:50. Then they turn to `h200`, whose
    front holds A 004's and A 002's seeds 0-1: at ~21:25, ~02:45, ~03:10 and ~08:05.
  - 26854533 and 26854534 (replacing 26824147 and 26824149 at 11:46) follow the two running H100
    workers (`afterany`). They read `record-003`, both A record queues (26854533 A 004 first,
    26854534 A 002 first), then `h200`. A new H200 job was estimated at 10-26 05:46.
- mit_normal_gpu: the two record chains above, pending.
- Cap: 2 normal + 4 preemptable = 6.
- I may update or cancel my own pending jobs without asking (INBOX 10-25).
- `run_item.sh` tries CUDA before each item and puts the item back on a CUDA error. For
  `records/v3/` items it also refuses a seed when the job has less time left than the budgets
  + 30 min (LOG 10-24).

## Queues

- `record-004_ladder_constrained_h200`, `record-002_fp32_skip_cosine`: seeds 2-4 each (an A seed is
  ~5.3 GPU-hours). `record-003_dm_math_fp32_h200`: seeds 2-4 (~6.2 h each).
- `h200` (7): A 004 seed 0, A 002 seed 0, A 004 seed 1, A 002 seed 1, then r12
  `a01_fp32_lr1e-6_skip_cos_advnone_h200` seeds 0-2 (r10's config on the H200). While the A record
  queues hold seeds, each wake keeps at least two A seeds ahead of r12. r12 stays if r10 seed 2
  (H100, from ~13:20) searches and holds past step 300; otherwise it comes out (LOG 09:19).
- `main` (3): r10 seed 2, r14 `a01_fp32_lr1e-6_skip_cos_mal0_tis2` seeds 0-1 (H100). `long`: empty.
- `h100` (4, read by no worker): r8 seed 1, d00_fp32_lr3e-6_skip seeds 0-2.
- `later` (26, read by no worker): parked H100 items (c90 seeds, r6, r8 seeds 3-4,
  d00_fp32_lr3e-6_skip seeds 3-4, a01_skip seeds 3-4, r9 seeds 1-2, r11 seeds 1-2).

## Best so far

- Track A, H100 dev:
  - A 002 and A 003 (top).
  - r10 (A 002 + mean-centred advantages) stops at two seeds: 0.340 and 0.180, against A 002's
    0.300 and 0.270. Seed 1 searched, peaked at 0.315 at step 300, then slid back to short
    outputs (LOG 05:18).
  - r8 (r5 + the -0.1 truncation penalty back from step 300) didn't hold r7's gains.
  - r9 (r5 + Clip-Cov 0.1 %, skipping off): seed 0 diverged at step ~300 (entropy 6, 98 %
    malformed); stopped. Clip-Cov is closed for track A (LOG 02:28).
- Track D, H100 dev: D 002's recipe averaged 0.153 over five research screens and 0.160 over its
  record seeds. dm51 (51 % DeepMind share) ended at 0.155 and 0.178 on seeds 0-1, against dm26's 0.152 and
  0.143 (paired mean +0.019). Seed 2 is queued. r11 at lr 1e-5 is behind dm26 at 156 min (0.105).

## Next

- Records: D 003 seeds 0-1 end ~14:30-15:05 (test-ladder scores in their result.json). A 004
  seed 0 starts at ~21:25 on a preemptable H200; the mit_normal_gpu chains are planned for 10-27
  10:00.
- H200 screens: r12 seeds 0-2 after the A record seeds. Track D H200 screens can start from D 003's
  saved pretrained models once seeds finish: r11's LR sweep (1e-5, 3e-5) as RL-only screens
  paired with D 003's seeds.
- H100 results still to come: dm51 seed 2 (~13:20), r13 seed 1 (~13:40), r10 seed 2 and r14 seed 0
  (~18:30). r14 tests whether r5's late collapse comes from the sampler/trainer mismatch: r5's seeds
  0-1 collapsed at steps 375-425 (~150-170 min), so r14 seed 0 shows it by ~16:30 (LOG 10:39).
- New options (commit 92238ab, off by default): `log_mismatch` logs the vLLM/trainer log-prob
  mismatch per step; `tis_cap` adds truncated importance weights.
- Closed today: r10 at two seeds (unless seed 2 changes the picture), r11 (higher D learning
  rates), W-REINFORCE (c90_pos01), KL 0.05 with mean-centred advantages (r13 holds the policy at
  the base; LOG 09:19).
- B200: try one short item first if H200 capacity runs short (INBOX 10-25).

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q`. Without it, the two-rank tests in
  `tests/test_distributed.py` fail on the login node ("libgomp: Thread creation failed").
- Eval arms: a dir with `eval.json`, queued like a screen, runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`, `dev_ladder_eval.py`). `summarize_screens.py` prints
  them.
- Screens that save weights write their logs to the pool weights dir
  (`~/orcd/pool/rlvr-ckpts/<arm>/seeds/<seed>/`) while they run. The repo seed dir gets them at the
  end.
- Track D pretraining is time-budgeted and its speed varies by node: 224k tokens/s on node4109
  against 359-373k in D 000's record. Read D results with their tokens/s. On the H200, pretraining
  reads more tokens in the same 112 min.
- Ubuntu 24 nodes (8 of ~35 H100 nodes on mit_preemptable, e.g. node2700): the worker adds gcc's
  multiarch paths. `module load` fails there.
