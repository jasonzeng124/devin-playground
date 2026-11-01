# Track A 005: fp32 GRPO at lr 1e-6 on a cosine, stopped at step 400

A track A record attempt on the H200 board. Mean test ladder score of all five seeds:
**0.3400 ± 0.0815**. Every seed finished within the 5-h budget.

`scripts/validate_record_v3.py` passes. Against the H200 holder, A 004 (0.2671 ± 0.0565), the
one-sided Welch test gives t = 1.64, df 7.1, p = 0.072. That is not significant at p < 0.05, so
A 005 does not replace A 004 as the track A holder.

## What ran

The recipe is A 001's, with these changes:

- Qwen3-1.7B-Base.
- Training puzzles come from the ladder-level generators. The dev and test ladders are excluded.
- Answers are constrained to well-formed expressions during rollouts.
- fp32 master weights.
- lr 1e-6 on a cosine down to 0.1x, counted over 500 steps (`budget_steps` 500).
- Zero-advantage rows are skipped.
- Truncated outputs get a training reward of 0.
- Training stops at step 400 (`max_steps` 400). `budget_steps_time_stop` would also stop it at
  17,400 s of real time.
- Sampler: `constrain: true`, T = 1, no think budget.
- No external data.

It was promoted from research screen `calibration/research_r16/mal0_st500`, which ran the same
schedule to step 500. That screen's five seeds averaged 0.397 ± 0.011 on the dev ladder at step 400
(LOG 2026-10-28 23:10).

The seeds ran from 2026-10-31 to 2026-11-01 on mit_preemptable H200 SXM workers (`NVIDIA H200`) on
MIT Engaging. Each seed ran `scripts/research/run_item.sh`: first `scripts/warm_start.sh`, untimed,
then `scripts/run_seed_v3.sh`. The software is this repository at the commits listed in
`docs/research/LOG.md`, with `uv.lock`'s package versions.

| seed | node | test ladder | pass@16 | mean tokens | truncated | in-loop final dev | steps | total_s | within 18,000 s | warm start |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | node4300 | 0.3995 | 0.6346 | 709 | 0.53 | 0.375 | 400 | 7781 | yes | 35 s |
| 1 | node4400 | 0.4076 | 0.6608 | 723 | 0.55 | 0.403 | 400 | 8867 | yes | 10 s |
| 2 | node4500 | 0.3821 | 0.6250 | 756 | 0.54 | 0.342 | 400 | 8775 | yes | 56 s |
| 3 | node4801 | 0.2209 | 0.4738 | 403 | 0.14 | 0.197 | 400 | 7618 | yes | 24 s |
| 4 | node4300 | 0.2899 | 0.5875 | 554 | 0.23 | 0.248 | 400 | 8075 | yes | 17 s |

- Test ladder score, mean of all five seeds: 0.3400 ± 0.0815. The in-loop final dev mean is 0.313.
- On the clock, the seeds took 127.0-147.8 min of the 300, because training stops at step 400.
- `warm_start.log` lists every warm start, including those before the preempted attempts.

## Seeds 3 and 4 ended lower

Seeds 0-2 kept long outputs: mean tokens 709-756, and about 54 % hit the 1,024-token limit. They
scored 0.38-0.41. Seeds 3 and 4 shortened their outputs, to 403 and 554 mean tokens and 14 % and
23 % truncated. Their scores dropped by every level, most at L4-L5, and they scored 0.22 and 0.29.
In the r16 screen, all five seeds were within 0.02 of each other at step 400, so the screen
understated the spread between seeds. Both seeds count like every other.

## Preempted attempts

Four attempts were preempted. They are kept in `extra/`, each with a `note.txt`.

- `extra/interrupted_0_1`: seed 0, at step 165 of 400, on node2500, which ran about 3x slow.
- `extra/interrupted_0_2`: seed 0, at step 1.
- `extra/interrupted_2_1`: seed 2, after all 400 steps, during the test ladder, before any test
  score was written.
- `extra/interrupted_3_1`: seed 3, at step 130.

Each was an infrastructure stop, so the same seed was rerun, as RULES_v3 allows ("Seeds and
statistics"). No seed was dropped or rerun for its score.
