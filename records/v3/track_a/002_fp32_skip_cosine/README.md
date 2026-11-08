# Track A 002: A 001 with fp32 master weights, cosine decay and zero-advantage rows skipped

A track A record attempt on the H200 board (RULES_v3 v3.1). Mean test ladder score of all five
seeds: **0.2784 ± 0.0904**, and every seed finished within the 5-h budget. The holder is A 004
(0.2671 ± 0.0565); see "Against the holder" for the Welch test.

## What ran

`config.json` changes A 001's recipe (`records/v3/track_a/001_ladder_constrained`) in four ways.
It comes from research screen `calibration/research_r2/a01_fp32_lr1e-6_skip_cos`:

- fp32 master weights (`master_fp32: true`).
- lr 1e-6, decaying on a cosine to 1e-7 over the budget (`lr_schedule: cosine`, `lr_final_frac: 0.1`).
- Groups where every rollout has the same reward are skipped (`skip_zero_advantage: true`).
- Scored with `constrain: true` at temperature 0.5, with no think budget.

The rest is A 001's recipe:

- Qwen3-1.7B-Base, GRPO for 290 min.
- Training puzzles come from the ladder-level generators. The dev and test ladders are excluded.
- Answers are constrained to well-formed expressions during rollouts.
- No external data.

The seeds ran from 2026-11-02 to 2026-11-08 on H200 SXM workers (`NVIDIA H200`), all on
mit_preemptable on MIT Engaging. Each seed ran `scripts/research/run_item.sh`: first
`scripts/warm_start.sh`, untimed, then `scripts/run_seed_v3.sh`. The software is this repository
at the commits listed in `docs/research/LOG.md`, with `uv.lock`'s package versions.

| seed | node | test ladder | pass@16 | mean tokens | in-loop final dev | steps | total_s | within 18,000 s | warm start |
|---|---|---|---|---|---|---|---|---|---|
| 0 | node4500 | 0.2518 | 0.2796 | 117 | 0.243 | 5485 | 17451 | yes | 28 s |
| 1 | node4802 | 0.2149 | 0.2863 | 190 | 0.210 | 2431 | 17554 | yes | 52 s |
| 2 | node4301 | 0.2809 | 0.3083 | 64 | 0.265 | 9020 | 17448 | yes | 11 s |
| 3 | node4301 | 0.2123 | 0.3500 | 271 | 0.183 | 1345 | 17448 | yes | 41 s |
| 4 | node4301 | 0.4320 | 0.5821 | 509 | 0.417 | 1595 | 17468 | yes | 77 s |

- Test ladder score, mean of all five seeds: 0.2784 ± 0.0904. The in-loop final dev mean is 0.264.
- On the clock, the seeds took 290.8-292.6 min of the 300.
- `warm_start.log` lists every warm start, including those before the preempted attempts.

## Answer length decides the seed

Four of the five seeds shortened their answers (64-271 mean tokens) and ran more steps. They scored
0.212-0.281, with pass@16 of only 0.28-0.35 (A 004's long-answer seeds reach 0.54-0.58). Seed 4 kept long answers (509 tokens),
searched, and scored 0.432 with pass@16 0.58. The recipe's mean is close to A 004's, but the
spread between seeds is wider. Every seed counts.

## Preempted attempts

Seven attempts were preempted. They are kept in `extra/`, each with a `note.txt`.

- `extra/interrupted_0_1`: seed 0, at 99 % of its training budget.
- `extra/interrupted_1_1`: seed 1, at 70 %.
- `extra/interrupted_1_2`: seed 1, at 4 %.
- `extra/interrupted_4_1`: seed 4, at 17 %.
- `extra/interrupted_4_2`: seed 4, at 36 %.
- `extra/interrupted_4_3`: seed 4, during model loading (0 %).
- `extra/interrupted_4_4`: seed 4, at 27 %.

All seven were stopped before the final eval and the test ladder. Each was an infrastructure stop,
so the same seed was rerun, as RULES_v3 allows ("Seeds and statistics"). No seed was dropped or
rerun for its score.

## Against the holder

`scripts/validate_record_v3.py records/v3/track_a/002_fp32_skip_cosine --compare-to
records/v3/track_a/004_ladder_constrained_h200`: all five seeds are valid. Mean 0.2784 against
A 004's 0.2671, Welch t = 0.24, df = 6.7, one-sided p = 0.41. **Not significant, so A 002 does not
replace A 004** and is not added to the leaderboard. It is kept as a valid, non-winning attempt.
