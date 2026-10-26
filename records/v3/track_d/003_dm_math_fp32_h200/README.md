# Track D 003: D 002's config on the H200 (failed: two seeds over the 6-h budget)

**Not a record.** Seeds 0 and 2 ran past the 6-h budget, so `validate_record_v3.py` fails the
record. All five seeds are kept here as they ran, and none was rerun or dropped. The maintainer's
ruling (2026-10-26): report D 003 as failed, and rerun D 002's config unchanged, with the untimed
warm-up that RULES_v3 now allows (v3.2), as `004_dm_math_fp32_h200_warm`.

## What ran

`config.json` is `records/v3/track_d/002_dm_math_fp32/config.json`, byte for byte. It was the first
H200 rerun of the H100 track D holder (RULES_v3 v3.1). See 002's README for the recipe: DeepMind
arithmetic in the 112-min pretraining, then 234 min of fp32 RL with zero-advantage rows skipped.
Sampler: `constrain: true`, T = 1, no think budget.

The seeds ran from 2026-10-25 to 2026-10-26 on mit_preemptable H200 SXM workers on Engaging
(`scripts/research/run_item.sh`, then `STAGE=all scripts/run_seed_v3.sh`), without a warm-up.

| seed | node | test ladder | pass@16 | in-loop final dev | pretrain_s | total_s | within 21,600 s |
|---|---|---|---|---|---|---|---|
| 0 | node4900 | 0.1590 | 0.2025 | 0.168 | 6876 | 21727 | **no** |
| 1 | node4301 | 0.1385 | 0.1854 | 0.125 | 6779 | 20906 | yes |
| 2 | node4300 | 0.1455 | 0.1808 | 0.173 | 7206 | 21724 | **no** |
| 3 | node4300 | 0.1734 | 0.2396 | 0.183 | 6734 | 20921 | yes |
| 4 | node4300 | 0.1526 | 0.2092 | 0.153 | 6732 | 20922 | yes |

Mean test score of all five: 0.1538 ± 0.0134. D 002 on the H100 scored 0.1512 ± 0.0202.

`extra/interrupted_1_1` and `extra/interrupted_1_2` hold seed 1's two preempted attempts. The first
was preempted 88.5 min into RL, the second during pretraining. Both were infrastructure stops, so
the seed was rerun under the same seed, as RULES_v3 allows (`note.txt` in each).

## Why seeds 0 and 2 ran over

The trainers' own clocks are fixed: 112 min of pretraining plus 14,040 s of RL, 20,760 s in all,
which leaves 840 s for start-up and saves. On the H100, D 002's seeds used 124-211 s of that. Here:

- Seeds 1, 3 and 4 used 146-162 s, as on the H100.
- Seed 0 used 967 s. It ran on node4900, whose CPUs were all allocated to other jobs. Its RL
  sampled at half speed, and its GRPO process took 811 s beyond its training clock. node4900 has
  been excluded from all workers since.
- Seed 2 used 964 s: 486 s extra in the pretraining stage and 478 s in the RL stage. The same node
  started seeds 3 and 4 later with normal times.

The excess was process start-up before each trainer's clock (imports, model creation, vLLM's
start-up), with the Python environment read cold over the network filesystems. During those days,
the workers' 150-s CUDA check timed out on 15 of 42 starts. `docs/research/LOG.md` (2026-10-26
06:40 and 11:45) has the details. The untimed warm start (`scripts/warm_start.sh`) reads the
environment into the page cache before the clock starts. With it, screens on the same nodes had vLLM
running 30-55 s after launch.
