# Status

(maintained by the research agent; see AGENT.md)

**v3.1 (2026-10-25): the reference GPU is the H200 SXM (`NVIDIA H200`). The H100 records keep a
separate board. The H200 board starts with three record runs, all queued (LOG 01:24):**

| record | config | runs on | first start (estimate) |
|---|---|---|---|
| `records/v3/track_a/004_ladder_constrained_h200` | A 001's, unchanged | mit_normal_gpu chain 1 (26824033-37) | 10-28 00:30 (Slurm's plan at 03:29) |
| `records/v3/track_a/002_fp32_skip_cosine` | as promoted 10-23 | mit_normal_gpu chain 2 (26824038-42) | 10-28 01:00 |
| `records/v3/track_d/003_dm_math_fp32_h200` | D 002's, unchanged | mit_preemptable H200 (26824147-50) | a new H200 job: 10-25 21:48 (estimate) |

- A 004 and A 002 run side by side, one seed per 6-h job. Chain 1 serves A 004 first and chain 2
  serves A 002 first; each helps the other when its own queue is empty. A 002 is compared with
  A 004 once both land.
- A 003 (`records/v3/track_a/003_fp32_mal0_s300`) waits, as the laptop asked.
- QUESTIONS.md (10-25 02:28) asks whether track A's H200 record seeds may also run on
  mit_preemptable, since a preemptable H200 job is estimated two days sooner. Until then they
  stay on mit_normal_gpu.
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

Updated 2026-10-25 03:31 EDT.

## Running

- mit_preemptable H100 workers (`main:long`):
  - 26786843 (node2700): r11 lr 1e-5 seed 0, since 02:36 (after dm51 seed 0, done 02:36).
  - 26803673 (node2905): dm51 seed 1, resumed 01:48, ends ~05:00.
  - 26786872 (node2905): r10 seed 1, resumed 01:43, ends ~04:40.
  - Whichever of the last two finishes first takes r11 lr 3e-5 seed 0 from `main`; the other
    exits, and its H200 successor waits for a GPU.
  - 26786844 (r9 seed 0, diverged) was cancelled at 02:27.
- H200 successors on mit_preemptable (`record-003_dm_math_fp32_h200:h200`, `--time-min=06:30:00`,
  RESERVE_MIN 380, NVL nodes excluded), one `afterany` on each H100 worker:
  - 26824148 is pending (Resources) since 02:27.
  - 26824147, 26824149 and 26824150 follow their H100 workers.
- mit_normal_gpu: the two record chains above, pending.
- Cap: 2 normal + 4 preemptable = 6.
- I may update or cancel my own pending jobs without asking (INBOX 10-25).
- `run_item.sh` tries CUDA before each item and puts the item back on a CUDA error. For
  `records/v3/` items it also refuses a seed when the job has less time left than the budgets
  + 30 min (LOG 10-24).

## Queues

- `record-004_ladder_constrained_h200`, `record-002_fp32_skip_cosine`, `record-003_dm_math_fp32_h200`:
  seeds 0-4 each (A: ~27 GPU-hours each; D: ~31).
- `h200` (3): r12 `a01_fp32_lr1e-6_skip_cos_advnone_h200` seeds 0-2. This is r10's config on the
  H200, paired with A 002's H200 record seeds.
- `main` (2): r11 dm26_fp32_lr1e-5_skip and dm26_fp32_lr3e-5_skip seed 0 (H100). `long`: empty.
- `h100` (13, read by no worker): the rest of the H100 screen queue (LOG 01:24, 02:28):
  - r10 seed 2;
  - r11 seeds 1-2 of both rates;
  - dm51 seed 2;
  - r8 seed 1;
  - d00_fp32_lr3e-6_skip seeds 0-2;
  - c90_pos01.
- `later` (22, read by no worker): parked H100 items (c90 seeds, r6, r8 seeds 3-4,
  d00_fp32_lr3e-6_skip seeds 3-4, a01_skip seeds 3-4, r9 seeds 1-2).

## Best so far

- Track A, H100 dev:
  - A 002 and A 003 (top).
  - r10 (A 002 + mean-centred advantages): seed 0 searched for the whole run and ended at 0.340.
    A 002's seed 0 ended at 0.300, one-guess. Seed 1 is at 0.285 at 140 min (A 002's seed 1
    ended at 0.270).
  - r8 (r5 + the -0.1 truncation penalty back from step 300) didn't hold r7's gains.
  - r9 (r5 + Clip-Cov 0.1 %, skipping off): seed 0 diverged at step ~300 (entropy 6, 98 %
    malformed); stopped. Clip-Cov is closed for track A (LOG 02:28).
- Track D, H100 dev: D 002's recipe averaged 0.153 over five research screens and 0.160 over its
  record seeds. dm51 (51 % DeepMind share) seed 0 ended at 0.155 (best 0.167), against dm26 seed 0's 0.152;
  seed 1 is at 0.127 at 145 min.

## Next

- Records: watch the chains start (estimate 10-26 21:44) and the D 003 workers take over from the
  H100 workers.
- H200 screens: r12 seeds 0-2 after D 003's seeds. Track D H200 screens can start from D 003's
  saved pretrained models once seeds finish: r11's LR sweep (1e-5, 3e-5) as RL-only screens
  paired with D 003's seeds.
- H100 results still to come: dm51 seeds 0-1, r9 seed 0, r10 seed 1. Log them as they end.
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
