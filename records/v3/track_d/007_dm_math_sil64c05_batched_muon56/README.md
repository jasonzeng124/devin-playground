# Track D 007: D 006 with 56 min of Muon pretraining, batched masking and 56 min more RL

D 006's recipe (`records/v3/track_d/006_dm_math_fp32_shape05_lr6e-6_sil64c05`) with the time moved towards RL:
pretraining uses **Muon for 56 min** instead of AdamW for 112, the RL phase gets the 56 min saved (234 -> 290 min),
and the constrained sampler applies its mask to the whole batch at once. Mean test ladder score of all five seeds:
**0.2644 ± 0.0076**, against D 006's 0.2444 ± 0.0158. Every seed finished within the 6-h budget.

`scripts/validate_record_v3.py` passes. Against the H200 holder, D 006 (0.2444 ± 0.0158), the one-sided Welch
test gives t = 2.55, df 5.7, p = 0.0226, so D 007 replaces D 006 as the track D holder. The gain is +0.020.

## What ran

`config.json` is D 006's config with these changes:

- `v3.pretrain.args` adds `--optimizer muon --lr 4e-3`, and `v3.pretrain.minutes` goes from 112 to 56 (Muon is
  already in `scripts/pretrain_scratch.py`; D 001 used the same setting).
- `time_budget_s`: 14,040 -> 17,400 (the RL phase, 234 -> 290 min). Pretraining plus RL is 20,760 s in both records.
- `constrain_batched: true`: the answer mask of constrained decoding is applied to the whole batch in one go
  instead of request by request (`rlvr_speedrun/grpo.py`, off by default). The mask is the same; sampling is faster,
  so the same RL time holds more steps.
- `description`.

Everything else is D 006 (and so D 005, D 004 and D 002):

- Pretraining from scratch on the pinned FineWeb-Edu shards, plus published DeepMind mathematics_dataset
  arithmetic (question-answer pairs, as published). See D 002's README for every external dataset with its
  revision.
- fp32 GRPO at constant lr 6e-6 with partial credit 0.5 (`shaping`), with zero-advantage rows skipped.
- Self-imitation as in D 006 (`sil_rows: 64`, `sil_coef: 0.5`, `sil_buffer: 4096`, `sil_by_level: true`), under
  the limits that D 006's README lists (`docs/research/INBOX.md` 2026-11-10 14:50 and 2026-11-12 11:56): only
  this run's own verifier-accepted completions, a buffer in memory that starts empty with every seed, nothing
  relabelled.
- Sampler: `constrain: true`, T = 1, no think budget.

The gating screen was `calibration/research_r81/d006_batched_muon56_rl17400`: this config with `no_save`, run end
to end on the H200 in the 3-seed form. Its in-loop final dev scores were 0.262 / 0.268 / 0.275, each at or above
D's line 0.259 (D 006's dev mean 0.229 + 0.03) (`docs/research/LOG.md`, 2026-11-16 13:53).

Seeds 0-3 ran on 2026-11-16 from 14:10 to 20:05 EST, and seed 4 from 2026-11-18 22:44 to 2026-11-19 04:36 EST,
on mit_preemptable H200 SXM workers (`NVIDIA H200`) on MIT Engaging. Each seed ran `scripts/research/run_item.sh`:
first `scripts/warm_start.sh`, untimed, then `STAGE=all scripts/run_seed_v3.sh`. The software is this repository
at the commits listed in `docs/research/LOG.md`, with `uv.lock`'s package versions. No code differs from the
repository. Seed 4 ran two days after the others, on a later commit: the commits in between add options that are
off by default (the native sampler, `reuse_sampler_allowed`, `share_group_prompt`, `time_update_parts`, fused
pretraining), and this config sets none of them.

| seed | node | test ladder | pass@16 | in-loop final dev | RL steps | pretrain_s | total_s | within 21,600 s | warm start |
|---|---|---|---|---|---|---|---|---|---|
| 0 | node4901 | 0.2606 | 0.3571 | 0.245 | 17,359 | 3370 | 20880 | yes | 41 s |
| 1 | node4901 | 0.2538 | 0.3454 | 0.252 | 19,641 | 3370 | 20879 | yes | 41 s |
| 2 | node4602 | 0.2646 | 0.3525 | 0.250 | 18,335 | 3369 | 20879 | yes | 25 s |
| 3 | node4801 | 0.2726 | 0.3542 | 0.255 | 19,735 | 3369 | 20875 | yes | 11 s |
| 4 | node4802 | 0.2703 | 0.3621 | 0.253 | 16,318 | 3377 | 20856 | yes | 50 s |

- Test ladder score, mean of all five seeds: 0.2644 ± 0.0076 (D 006: 0.2444 ± 0.0158). Paired by seed with
  D 006 the differences are +0.0354 / +0.0051 / -0.0016 / +0.0391 / +0.0218. The in-loop final dev mean is
  0.251, and the mean pass@16 is 0.3543.
- On the clock, the seeds took 347.6-348.0 min of the 360.
- `warm_start.log` lists every warm start (seed 4 has two lines, one for each attempt).
- No seed was dropped. Seed 4's first attempt was preempted and the same seed reran (next section).

## Preempted attempt

Seed 4's first attempt (node4802, started 2026-11-17 16:04Z after a 216 s warm start) was preempted at about
18:19Z, after its pretraining, at RL step 3,917 (4,550 s of the 17,400-s RL budget). Its logs are in
`extra/interrupted_4_1`, with a note. As RULES_v3 allows, the preemption reran the same seed from scratch (it
waited in the H200 queue until 2026-11-19 03:45Z), and the seed counts once, with its complete run. No other
seed was interrupted.

## Where the gain comes from

The record is the test ladder comparison above. The screens around it say which of the three changes earns it,
and they do not support the Muon split by itself (`docs/research/LOG.md`, 2026-11-19 02:05 and 05:26):

- D 007's seeds ran 16,318 to 19,735 RL steps against D 006's 10,068 to 11,380 (1.05 against 0.77 steps a second,
  over an RL phase 24 % longer).
- `calibration/research_r77/d006_batched_h200` is D 006 with `constrain_batched` alone (H200, timed): in-loop
  final dev 0.253 / 0.218 / 0.273, mean 0.248, against D 006's 0.229.
- The gating screen r81 was +0.020 over r77 with a standard error of 0.0165 (three unpaired seeds each), which
  is not a firm margin. This record's five seeds have an in-loop dev mean of 0.251, r77's level.
- r86 (step-capped pairs on an H100, 2026-11-18): 56 min of Muon with 24 % more RL steps against 112 min of
  AdamW came out at -0.011 on dev over five pairs (standard error 0.010).

So on dev, 56 min of Muon plus 56 min more RL is about level with 112 min of AdamW, and most of D 007's gain
over D 006 comes from the extra RL steps that batched masking gives. The gating screen's three seeds were on the
high side of what the config does.
