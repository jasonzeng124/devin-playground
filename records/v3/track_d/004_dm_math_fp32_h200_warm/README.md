# Track D 004: D 002's config on the H200, with the untimed warm-up

The first track D record on the H200 board (RULES_v3 v3.1). It reruns the H100 holder's config
(`records/v3/track_d/002_dm_math_fp32`) unchanged. Each seed runs after the untimed warm-up that
RULES_v3 allows since v3.2 ("Untimed warm-up"). Mean test ladder score of all five seeds:
**0.1552 ± 0.0217**, and every seed finished within the 6-h budget.

## What ran

`config.json` is `records/v3/track_d/002_dm_math_fp32/config.json`, byte for byte. See 002's README
for the recipe and for every external dataset with its revision. In short:

- 112 min of pretraining from scratch on the pinned FineWeb-Edu shards, plus published DeepMind
  mathematics_dataset arithmetic (question-answer pairs, as published).
- Then 234 min of fp32 GRPO, with zero-advantage rows skipped.
- Sampler: `constrain: true`, T = 1, no think budget.

D 003, the first H200 rerun, failed because two seeds ran past the budget. The maintainer's ruling
(2026-10-26) was to rerun the same config with the warm-up as this record.

The seeds ran from 2026-10-27 to 2026-10-30 on mit_preemptable H200 SXM workers (`NVIDIA H200`) on
MIT Engaging. Each seed ran `scripts/research/run_item.sh`: first `scripts/warm_start.sh`, untimed,
then `STAGE=all scripts/run_seed_v3.sh`. The software is this repository at the commits listed in
`docs/research/LOG.md`, with `uv.lock`'s package versions.

| seed | node | test ladder | pass@16 | in-loop final dev | pretrain_s | total_s | within 21,600 s | warm start |
|---|---|---|---|---|---|---|---|---|
| 0 | node4300 | 0.1832 | 0.2367 | 0.182 | 6731 | 20816 | yes | 36 s |
| 1 | node4400 | 0.1296 | 0.1604 | 0.130 | 6739 | 20993 | yes | 57 s |
| 2 | node4300 | 0.1384 | 0.2100 | 0.137 | 6728 | 20892 | yes | 144 s |
| 3 | node2500 | 0.1677 | 0.2283 | 0.183 | 6736 | 20941 | yes | 59 s |
| 4 | node2500 | 0.1572 | 0.2037 | 0.163 | 6736 | 20932 | yes | 11 s |

- Test ladder score, mean of all five seeds: 0.1552 ± 0.0217. The in-loop final dev mean is 0.159.
- On the clock, the seeds took 346.9-349.9 min of the 360. The trainers' own clocks take
  20,760 s, so start-up and saves took 56-233 s per seed. In D 003, without the warm-up, two seeds
  took about 965 s.
- `warm_start.log` lists every warm start, including those before the preempted attempts.
- For reference only (a different GPU, not compared on the board): D 002 scored 0.1512 ± 0.0202 on
  the H100, and D 003 0.1538 ± 0.0134 on the H200.

## Preempted attempts

Three attempts were preempted. They are kept in `extra/`, each with a `note.txt`.

- `extra/interrupted_0_1`: seed 0, in the first minutes of pretraining.
- `extra/interrupted_0_2`: seed 0, after the full pretraining and 3,644 s of RL.
- `extra/interrupted_1_1`: seed 1, 64.6 min into pretraining.

Each was an infrastructure stop, so the same seed was rerun, as RULES_v3 allows ("Seeds and
statistics"). No seed was dropped or rerun for its score.
