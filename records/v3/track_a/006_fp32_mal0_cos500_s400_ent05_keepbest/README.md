# Track A 006: A 005 with an entropy floor, keeping its best in-loop dev checkpoint

A track A record on the H200 board. Mean test ladder score of all five seeds:
**0.4123 ± 0.0127**. Every seed finished within the 5-h budget.

`scripts/validate_record_v3.py` passes. Against the H200 holder, A 004 (0.2671 ± 0.0565), the
one-sided Welch test gives t = 5.61, df 4.4, p = 0.0019, so A 006 replaces A 004 as the track A
holder. For reference, against A 005 (0.3400 ± 0.0815, not a holder) it gives t = 1.96, df 4.2,
p = 0.059.

## What ran

The recipe is A 005's, with two changes:

- An entropy floor: an entropy bonus with coefficient 0.05 applies while the policy's entropy is
  below 0.1 (`entropy_coef` 0.05, `entropy_target` 0.1).
- `keep_best_dev`: the run evaluates on the dev ladder every 50 steps from step 50, and ends on the
  weights of its best in-loop dev eval. Those evals and the weight copies are on the clock. **This
  record chose its checkpoint by its own in-loop dev evals**, which RULES_v3 allows since v3.4
  (2026-11-10). The test ladder is scored once, on the chosen weights.

A 005's recipe, unchanged:

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

It was promoted from research screen `calibration/research_r58/a005_ent05_keepbest_h200` (H200),
which is this config. Its kept dev evals were 0.363, 0.398 and 0.330, each at or above the 0.305
bar set before the screen ran (A 004's dev line 0.285, plus 0.02 because the kept eval is the best
of eight). Seed 2 was promoted at step 300, while still running (LOG 2026-11-12 00:11).

The seeds ran on 2026-11-12 on mit_preemptable H200 SXM workers (`NVIDIA H200`) on MIT Engaging.
Each seed ran `scripts/research/run_item.sh`: first `scripts/warm_start.sh`, untimed, then
`scripts/run_seed_v3.sh`. The software is this repository at the commits listed in
`docs/research/LOG.md`, with `uv.lock`'s package versions.

| seed | node | test ladder | pass@16 | mean tokens | truncated | kept step | kept in-loop dev | steps | total_s | within 18,000 s | warm start |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | node4601 | 0.4099 | 0.6750 | 703 | 0.53 | 400 | 0.397 | 400 | 8464 | yes | 19 s |
| 1 | node4801 | 0.3932 | 0.6825 | 726 | 0.54 | 350 | 0.375 | 400 | 8722 | yes | 87 s |
| 2 | node4601 | 0.4148 | 0.6879 | 718 | 0.53 | 400 | 0.397 | 400 | 8315 | yes | 8 s |
| 3 | node4601 | 0.4149 | 0.6763 | 712 | 0.54 | 400 | 0.392 | 400 | 9018 | yes | 10 s |
| 4 | node4801 | 0.4285 | 0.6846 | 697 | 0.52 | 400 | 0.380 | 400 | 9586 | yes | 61 s |

- Test ladder score, mean of all five seeds: 0.4123 ± 0.0127. The kept in-loop dev mean is 0.388.
- On the clock, the seeds took 138.6-159.8 min of the 300, because training stops at step 400.
- `warm_start.log` lists every warm start.

## What the checkpoint choice did

Four seeds kept their last checkpoint (step 400). Seed 1 kept step 350, where its dev eval was
0.375 against 0.365 at step 400. So on this run the option changed one seed's weights, by one eval
interval.

## No seed shortened its outputs

In A 005, two seeds of five shortened their outputs (403 and 554 mean tokens) and scored 0.22 and
0.29. Here all five kept long outputs (697-726 mean tokens, 52-54 % truncated) and scored
0.39-0.43, so the spread between seeds fell from 0.082 to 0.013. In the screens the entropy floor
stopped a slow late slide in entropy (LOG 2026-10-29, r21), but five seeds are too few to say it
is what held the outputs long here.

## Preempted attempts

None. Every seed ran once, and `extra/` does not exist.
