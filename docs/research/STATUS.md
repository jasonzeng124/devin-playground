# Status

(maintained by the research agent; see AGENT.md)

**v3.1 (2026-10-25): the reference GPU is the H200 SXM (`NVIDIA H200`). The H100 records keep a
separate board. The H200 board starts with three record runs (LOG 10-25 01:24):**

| record | config | runs on | state |
|---|---|---|---|
| `records/v3/track_d/003_dm_math_fp32_h200` | D 002's, unchanged | preemptable H200 workers | seed 0 done (test 0.1590, dev 0.168); seed 1 reruns when 26824150 restarts; seeds 2-4 queued |
| `records/v3/track_a/004_ladder_constrained_h200` | A 001's, unchanged | preemptable H200 workers and mit_normal_gpu chain 1 (26824033-37) | seeds 0-4 queued |
| `records/v3/track_a/002_fp32_skip_cosine` | as promoted 10-23 | preemptable H200 workers and mit_normal_gpu chain 2 (26824038-42) | seeds 0-4 queued |

- A preempted seed is an infrastructure stop: its partial attempt goes to `extra/` and the same seed
  reruns (INBOX 10-25 11:15).
- A 004 and A 002 run side by side; A 002 is compared with A 004 once both land. A 003
  (`records/v3/track_a/003_fp32_mal0_s300`) waits, as the laptop asked.
- Promotion uses the H100 lines until the reruns land: A 0.2703 (163 of 600 per seed for the
  3-seed form), D 0.190 (115 of 600). Then the lines come from the reruns' in-loop final dev (PLAN).
- node4900 is excluded from every worker: its CPUs are oversubscribed, and D's RL ran at half speed
  there (LOG 14:30).

H100 board (closed 10-24): A 001 holds track A (test 0.2481, dev 0.2403); D 002 holds track D
(test 0.1512 ± 0.0202, dev 0.160).

Promoted on H100 dev numbers, not yet recorded: A 002 (dev 0.324; seeds 0.300 / 0.270 / 0.472 /
0.315 / 0.263) and A 003 (dev 0.324; seeds 0.385 / 0.348 / 0.238 / 0.348 / 0.302; about one seed in
five of its recipe collapses between steps 250 and 300).

Updated 2026-10-25 23:05 EDT.

## AWS boxes: off

INBOX 10-25 22:00: the capacity blocks are under a refund review and must stay unused. No box
launches (no `run-instances` with the `rlvr-box-*` templates) and no AWS spending, including the
$500 emergency budget, until a later INBOX note says otherwise. The launch schedules are deleted
(checked read-only at 23:00: none, and no instances). `queue/b200.txt` (r15) is parked; its
`mal0_st500` arm runs on Engaging as r16.

## Running

- mit_preemptable H100 worker (`main:long`) 26803673 runs r14 seed 1 on node2700 since 18:22 (step
  378 at 22:53, dev 0.368 at step 350; ends ~23:45), then r16 seeds 0-2 (`main`, ~4.2 h each, to
  ~12:30). 26786843 finished r10 seed 2 and exited
  (18:42). Their H200 successors 26854533 (pending, eligible) and 26854534 (`afterany` 26803673) read
  `record-003`, both A record queues, then `h200`.
- H200 workers on mit_preemptable (`--time-min=06:30:00`, RESERVE_MIN 380, NVL nodes and node4900
  excluded): 26824150 (requeued at 12:10, pending; reruns D 003 seed 1 first), 26862717 (submitted 14:22 to replace node4900's 26824148; reads `record-003`, A 004's and A 002's
  queues, then `h200`). Slurm estimates new H200 starts at 10-26 ~07:45.
- mit_normal_gpu: the two record chains, pending (Slurm's plan: 10-27).
- Cap: 2 normal + 4 preemptable = 6. I may update or cancel my own pending jobs without asking.

## Queues

- `record-003_dm_math_fp32_h200`: seeds 2-4. `record-004_ladder_constrained_h200`,
  `record-002_fp32_skip_cosine`: seeds 2-4 each.
- `h200` (4): A 004 seed 0, A 002 seed 0, A 004 seed 1, A 002 seed 1 (r12 left at 18:46: r10 closed).
- `main` (3): r16 `mal0_st500` seeds 0-2. `b200` (32, r15), `h100`, `later`: parked, read by no worker.

## Best so far

- Track A, H100 dev: A 002 and A 003 (0.324). Best single screen seed: r14 seed 0 (r5 + TIS) 0.398
  (best 0.410), held by its slow sampling's annealed lr, not by TIS (LOG 18:46); r16 / r15
  `mal0_st500` test that. Closed: r10 (mean-centred advantages, 0.261 on three seeds), r9 (Clip-Cov),
  r13 (KL 0.05 with mean-centred advantages), W-REINFORCE, training temperature 1.2 in the -0.1 regime.
- Track D, H100 dev: D 002's recipe 0.153 (screens) / 0.160 (record seeds). Closed: dm51 (+0.010 over
  dm26 on three seeds), r11 (lr 1e-5, 3e-5).

## Next

- ~23:45: r14 seed 1 ends (second seed of the slow-clock condition, mismatch logged); r16 seed 0
  starts, the first GPU run of `budget_steps` (check its first steps, `clock_s` and lr).
- ~04:00, ~08:15, ~12:30: r16 seeds 0-2 end. r16 is step-capped, so its 3-seed screen can gate a
  record (PLAN: H200 or step-capped): every seed >= 0.2703 and none collapsed. A record config
  would also need a hard time stop next to `budget_steps` (a new option, off by default), so a run
  that searches slowly still stops at 290 min.
- If r16 fails: the next screens come from the parked r15 arms (lr 5e-7, truncation -0.5 in solved
  groups).
- Track D H200 screens can start from D 003's saved pretrained models once seeds finish.

## Notes

- Run tests as `OMP_NUM_THREADS=2 uv run pytest -q` (two-rank tests fail on the login node otherwise).
- Eval arms: a dir with `eval.json`, queued like a screen, runs the reference scorer on the dev
  ladder (`scripts/research/run_item.sh`). `summarize_screens.py` prints them.
- Step-capped runs (`budget_steps`) log `clock_s`; `summarize_screens.py` reads their minutes from it.
- Track D pretraining is time-budgeted and its speed varies by node; read D results with their
  tokens/s.
- Ubuntu 24 nodes: the worker adds gcc's multiarch paths; `module load` fails there.
