# Status

(maintained by the research agent; see AGENT.md)

**v3.1 (2026-10-25): the reference GPU is the H200 SXM (`NVIDIA H200`). The H100 records keep a
separate board. The H200 board starts with three record runs (LOG 10-25 01:24):**

| record | config | runs on | state |
|---|---|---|---|
| `records/v3/track_d/003_dm_math_fp32_h200` | D 002's, unchanged | preemptable H200 workers | seeds 0, 2 done (test 0.1590, 0.1455), both **over budget** by 124-127 s (LOG 06:40); seeds 1, 3, 4 running |
| `records/v3/track_a/004_ladder_constrained_h200` | A 001's, unchanged | none: **held** | seeds 0-4 in `queue/held_records.txt` |
| `records/v3/track_a/002_fp32_skip_cosine` | as promoted 10-23 | none: **held** | seeds 0-4 in `queue/held_records.txt` |

- **Start-up overrun (LOG 06:40, QUESTIONS 06:40).** On the H200 nodes, start-up and saves take
  ~16 min a D seed against 2-3.5 min on the H100. It looks like cold reads of `~/rlvr-venv` over
  NFS, and it pushes D 003's seeds past 21600 s. Track A's margin is only 600 s, so the A records
  are held until the maintainer answers: may records use an untimed warm-up (no GPU) before each
  timed stage, and how should D 003 be handled?
- A preempted seed is an infrastructure stop: its partial attempt goes to `extra/` and the same seed
  reruns (INBOX 10-25 11:15). D 003 has `extra/interrupted_1_1` and `interrupted_1_2`.
- A 004 and A 002 run side by side once released; A 002 is compared with A 004 once both land.
  A 003 (`records/v3/track_a/003_fp32_mal0_s300`) waits, as the laptop asked.
- Promotion uses the H100 lines until the reruns land: A 0.2703 (163 of 600 per seed for the
  3-seed form), D 0.190 (115 of 600). Then the lines come from the reruns' in-loop final dev (PLAN).
- node4900 is excluded from every worker: its CPUs are oversubscribed, and D's RL ran at half speed
  there (LOG 14:30). D 003 seed 0 ran there before the exclusion.

H100 board (closed 10-24): A 001 holds track A (test 0.2481, dev 0.2403); D 002 holds track D
(test 0.1512 ± 0.0202, dev 0.160).

Promoted on H100 dev numbers, not yet recorded: A 002 (dev 0.324; seeds 0.300 / 0.270 / 0.472 /
0.315 / 0.263) and A 003 (dev 0.324; seeds 0.385 / 0.348 / 0.238 / 0.348 / 0.302; about one seed in
five of its recipe collapses between steps 250 and 300).

Updated 2026-10-26 06:50 EDT.

## AWS boxes: off

INBOX 10-25 22:00: the capacity blocks are under a refund review and must stay unused. No box
launches (no `run-instances` with the `rlvr-box-*` templates) and no AWS spending, including the
$500 emergency budget, until a later INBOX note says otherwise. The launch schedules are deleted
(checked read-only at 23:00: none, and no instances). `queue/b200.txt` (r15) is parked; its
`mal0_st500` arm runs on Engaging as r16.

## Running

- mit_preemptable H100 worker (`main:long`) 26803673: r16 seed 0, preempted at 05:23 on node2904
  (step ~306), resumed on node2901 from the 05:09 save; ends ~07:40. Then r16 seed 1 (`main`). Its
  H200 successor 26854534 (`afterany` 26803673) reads `record-003`, both A record queues, then
  `h200`.
- H200 workers on mit_preemptable (`--time-min=06:30:00`, RESERVE_MIN 380, NVL nodes and node4900
  excluded), all running D 003 seeds:
  - 26854533: seed 3 on node4300 since 02:00, ends ~08:15.
  - 26824150: seed 1 on node4301 since 04:06, ends ~10:20.
  - 26862717: seed 4 on node4300 since 05:42, ends ~11:50.
  - Next they take r16 seeds 2-4 from `h200`.
- mit_normal_gpu: the two record chains (26824033-37, 26824038-42), pending; Slurm's plan is 10-27
  11:00 / 14:00. They read only the A record queues, which are empty while the records are held,
  so they would exit at once. Deal with them before then.
- Cap: 2 normal + 4 preemptable = 6. I may update or cancel my own pending jobs without asking.

## Queues

- `record-003_dm_math_fp32_h200`, `record-004_ladder_constrained_h200`, `record-002_fp32_skip_cosine`:
  empty. `queue/held_records.txt`: the ten A 004 / A 002 items, read by no worker.
- `h200` (3): r16 `mal0_st500` seeds 2-4. `main` (1): r16 seed 1.
- `b200` (32, r15), `h100`, `later`: parked, read by no worker.

## Best so far

- Track A, H100 dev: A 002 and A 003 (0.324). Best screen: r14 (r5 + TIS) 0.401 on two seeds
  (0.398, 0.403), where r5 fell to 0.087. TIS clipped nothing, so the likely cause is its slow
  sampling's earlier lr anneal by step (LOG 03:00). r16 tests that by step: seed 0 had dev 0.335 at
  step 200, 0.343 at 250 and 0.355 at 300 (r14 seed 1: 0.250, 0.287, 0.348). Closed: r10
  (mean-centred advantages, 0.261 on three seeds), r9 (Clip-Cov), r13 (KL 0.05 with mean-centred
  advantages), W-REINFORCE, training temperature 1.2 in the -0.1 regime.
- Track D: D 002's recipe, H100 dev 0.153 (screens) / 0.160 (record seeds). On the H200, D 003
  dev 0.168 / 0.173 and test 0.1590 / 0.1455 (seeds 0, 2). Closed: dm51 (+0.010 over dm26 on three
  seeds), r11 (lr 1e-5, 3e-5).

## Next

- Build the warm-up as an off-by-default option of `run_seed_v3.sh` and measure it on a compute
  node. Records use it only after the maintainer's answer (QUESTIONS 06:40).
- ~07:40: r16 seed 0 ends. Steps 375-525 are where r5's seeds collapsed. Then plan r17 on r16
  (e.g. truncation -0.5 in solved groups).
- ~15:30: all five r16 seeds done. r16 is step-capped, so its 5-seed screen can gate a record (PLAN:
  H200 or step-capped). The bar: mean >= 0.2703, at least 4 of 5 seeds above 0.240, at most one
  collapsed. A record from it would set `budget_steps_time_stop` (added 03:00) and needs the
  start-up fix.
- Track D H200 screens can start from D 003's saved pretrained models.

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q` (two-rank tests fail on the login node otherwise).
- Eval arms: a dir with `eval.json`, queued like a screen, runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`). `summarize_screens.py` prints them.
- Step-capped runs (`budget_steps`) log `clock_s`; `summarize_screens.py` reads their minutes from it.
  `budget_steps_time_stop` also stops them at `time_budget_s` (for records).
- Track D pretraining is time-budgeted and its speed varies by node; read D results with their
  tokens/s.
- Ubuntu 24 nodes: the worker adds gcc's multiarch paths; `module load` fails there.
- Record seeds: check `clock.json` `within_budget` as soon as a seed finishes.
