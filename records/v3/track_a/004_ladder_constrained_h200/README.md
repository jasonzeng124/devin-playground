# Track A 004: A 001's config on the H200, with the untimed warm-up

The first track A record on the H200 board (RULES_v3 v3.1). It reruns the H100 holder's config
(`records/v3/track_a/001_ladder_constrained`) unchanged. Each seed runs after the untimed warm-up
that RULES_v3 allows since v3.2 ("Untimed warm-up"). Mean test ladder score of all five seeds:
**0.2671 ± 0.0565**, and every seed finished within the 5-h budget.

## What ran

`config.json` is `records/v3/track_a/001_ladder_constrained/config.json`, byte for byte. See 001's
README for the recipe:

- Qwen3-1.7B-Base, GRPO for 290 min.
- Training puzzles come from the ladder-level generators. The dev and test ladders are excluded.
- Answers are constrained to well-formed expressions during rollouts.
- lr 4e-6.
- Sampler: `constrain: true`, T = 1, no think budget.
- No external data.

The seeds ran from 2026-10-29 to 2026-10-30 on H200 SXM workers (`NVIDIA H200`) on MIT Engaging.
Seeds 0-3 ran on mit_preemptable and seed 4 on mit_normal_gpu. Each seed ran
`scripts/research/run_item.sh`: first `scripts/warm_start.sh`, untimed, then
`scripts/run_seed_v3.sh`. The software is this repository at the commits listed in
`docs/research/LOG.md`, with `uv.lock`'s package versions.

| seed | node | test ladder | pass@16 | mean tokens | in-loop final dev | steps | total_s | within 18,000 s | warm start |
|---|---|---|---|---|---|---|---|---|---|
| 0 | node4301 | 0.3135 | 0.5754 | 525 | 0.287 | 998 | 17463 | yes | 31 s |
| 1 | node4301 | 0.2358 | 0.5579 | 425 | 0.230 | 1042 | 17438 | yes | 9 s |
| 2 | node4801 | 0.3132 | 0.5429 | 453 | 0.310 | 1018 | 17527 | yes | 71 s |
| 3 | node4801 | 0.2895 | 0.5704 | 543 | 0.263 | 921 | 17535 | yes | 71 s |
| 4 | node3100 | 0.1834 | 0.2429 | 47 | 0.185 | 2501 | 17508 | yes | 59 s |

- Test ladder score, mean of all five seeds: 0.2671 ± 0.0565. The in-loop final dev mean is 0.255.
- On the clock, the seeds took 290.6-292.3 min of the 300.
- `warm_start.log` lists every warm start, including those before the preempted attempts.
- For reference only (a different GPU, not compared on the board): A 001 scored 0.2481 ± 0.0460 on
  the H100.

## Seed 4 collapsed into single guesses

Seed 4 shortened its answers to about 47 tokens. It answers with one guess and no search, and it ran
2,501 steps against 921-1,042 for the others. It still scored 0.1834. A 001's seed 2 did the same on
the H100 (about 40 tokens, 0.186), so this recipe collapses on about one seed in five on either
GPU. The seed counts like every other.

## Preempted attempts

Five attempts were preempted. They are kept in `extra/`, each with a `note.txt`.

- `extra/interrupted_0_1`: seed 0, at 99 % of its training budget.
- `extra/interrupted_2_1`: seed 2, at 40 %.
- `extra/interrupted_2_2`: seed 2, at 59 %.
- `extra/interrupted_3_1`: seed 3, at 11 %.
- `extra/interrupted_3_2`: seed 3, at 7 %.

All five were stopped before the final eval and the test ladder. Each was an infrastructure stop,
so the same seed was rerun, as RULES_v3 allows ("Seeds and statistics"). No seed was dropped or
rerun for its score.
