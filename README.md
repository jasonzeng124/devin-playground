# RLVR Speedrun (Countdown)

A leaderboard for **reinforcement learning with verifiable rewards (RLVR)**, in the spirit of
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). Teach a language model to solve
Countdown puzzles (combine the given numbers with the allowed operators, each used once, to hit
a target) **as well as possible within a fixed GPU budget**. Training uses only puzzles and a
deterministic verifier: no solutions, no other models.

**Benchmark v3** (current) scores each entry on a frozen **difficulty ladder**: 6 levels x 400
puzzles, from 4 numbers with `+ -` up to 6 numbers with all four operators. The score is the mean
per-level pass@1 over 16 samples per puzzle. The rules are in [docs/RULES_v3.md](docs/RULES_v3.md);
the evidence behind each choice is in [docs/v3_design.md](docs/v3_design.md).

## Tracks

| track | start from | budget (1x H100 80GB SXM) | data |
|---|---|---|---|
| **A** | `Qwen/Qwen3-1.7B-Base` (pinned revision) | 5 h of training | generated puzzles + verifier verdicts |
| **D** (experimental) | random init, any architecture | 6 h **including pretraining** | FineWeb-Edu (pinned) for pretraining; RL as track A |

Track D is labelled experimental: every pilot plateaus at ~0.15-0.17 on the dev ladder, whatever
the optimizer, model size, pretraining length or exploration settings. The models learn a few
fixed answer templates and never do arithmetic. Breaking 0.20 is the open challenge.

## Leaderboard (v3)

### Track A

| # | record | test ladder score (mean ± sd) | N | clock | author |
|---|---|---:|--:|---|---|
| 000 | [000_baseline](records/v3/track_a/000_baseline) | 0.1295 ± 0.0098 | 5 | 290 min GRPO | @jasonzeng124 |

### Track D

| # | record | test ladder score (mean ± sd) | N | clock | author |
|---|---|---:|--:|---|---|
| 000 | [000_baseline](records/v3/track_d/000_baseline) | 0.1166 ± 0.0163 | 5 | 112 min pretraining + 234 min GRPO | @jasonzeng124 |

Not records: [track D 001_muon_pretrain](records/v3/track_d/001_muon_pretrain) (0.1162 ± 0.0156,
p = 0.52 against 000; a negative result).

A new record must beat the holder's mean with a one-sided Welch t-test (p < 0.05) over >= 5 seeds.

## Quickstart

```bash
uv sync                      # CPU torch by default; tests run on CPU
uv run pytest -q

# on a GPU box (reference software: vllm 0.30.0, torch 2.13.0+cu130, transformers 5.17.0)
pip install "vllm==0.30.0" "transformers==5.17.0" pyarrow numpy accelerate huggingface_hub && pip install -e . --no-deps
python scripts/prepare_track_d_data.py            # track D only: pinned FineWeb-Edu shards + tokenizer (CPU, untimed)

# one seed end to end: (pretraining,) GRPO, clock.json, test-ladder score with the record's sampler
bash scripts/run_seed_v3.sh records/v3/track_d/000_baseline 0

# cheap development loop on any GPU (4 min runs; unranked)
bash scripts/run_seed_v3.sh records/v3/smoke/track_d 0

# check a record before opening a PR
uv run python scripts/validate_record_v3.py records/v3/track_d/001_my_record --compare-to records/v3/track_d/000_baseline
```

`scripts/aws/spot_launch.sh` (`V3=1`) runs seeds on fresh AWS GPU instances that terminate
themselves and push results to S3. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to submit a
record.

## Layout

```
rlvr_speedrun/
  countdown.py       puzzle generator, prompt, verifier (part of the benchmark: do not change)
  ladder.py          the v3 difficulty ladder and its training stream
  ladder_confirm.py  the v3 scorer (test ladder, 16 samples, the record's sampler)
  constrained.py     answer-constrained decoding (vLLM logits processor + trainer renormalisation)
  grpo.py            the GRPO trainer (vLLM rollouts); config fields are CLI flags
  benchmark.py       protocol constants and pins
scripts/
  run_seed_v3.sh             one v3 seed end to end
  prepare_track_d_data.py    pinned track D data (CPU)
  pretrain_scratch.py        track D baseline pretraining (Llama-style, AdamW or Muon)
  validate_record_v3.py      v3 record checks + holder test (CI runs validate_all_records.py)
  aws/, slurm/               unattended runs on AWS and on a Slurm cluster
data/ladder_v3_{dev,test}.jsonl   frozen ladders (tune on dev; test is for the final score only)
records/v3/track_{a,d}/<NNN>_<slug>/   one folder per record
calibration/                 pilot and sweep logs behind the design decisions
```

## Earlier versions (retired)

v1 and v2 raced **Qwen2.5-0.5B** to a pass-rate threshold on 4-number puzzles (v1: 10 %, v2: 20 %
with a separate confirmation set). Their rules are in [RULES.md](RULES.md), and the v2
calibration is in [docs/plateau_calibration.md](docs/plateau_calibration.md). v3 replaced them
because one threshold can't fit very different models, and a ladder shows where the gains come
from. The v1 leaderboard is kept as history.

Hardware: **one H100 80GB SXM**, all seeds of a record on one provider. Track A
base model: **Qwen/Qwen2.5-0.5B** (pretrained-only checkpoint, no
post-training). Threshold: **20% pass rate** (v2; v1: 10%) on the frozen 2,000-puzzle eval,
N >= 3 consecutive seeds; a new record holder must beat the old one at
`p < 0.05` (one-sided Welch t-test, see [RULES.md](RULES.md)).

#### v1 Track A — fixed base model, RL only

| # | record | time to 10% (mean ± std) | N | final pass rate | guardrail | vs previous | author |
|---|--------|-------------------------:|--:|----------------:|-----------|-------------|--------|
| 1 | [002_curriculum_qwen05b_h100](records/track_a/002_curriculum_qwen05b_h100) | 783 ± 216 s | 3 | 0.104 | not measured | first record | Devin / @jasonzeng124 |
| 2 | [003_nokl_deferred_eval_h100](records/track_a/003_nokl_deferred_eval_h100) | 573 ± 214 s | 3 | 0.105 | not measured | untested (pre-dates the rule) | Devin / @jasonzeng124 |
| 3 | [004_nokl_guardrail_h100](records/track_a/004_nokl_guardrail_h100) | 958 ± 348 s ¹ | 10 | 0.106 | pass (Δloss +0.001) | same recipe as 003 | Devin / @jasonzeng124 |
| 4 | **[005_prefix_kv_compile_h100](records/track_a/005_prefix_kv_compile_h100)** (holder) | **460 ± 196 s** ¹ | 10 | 0.108 | pass (Δloss +0.001) | beats 004, p = 0.0007 (p = 0.0007 on the 7 vs 7 RunPod seeds) | Devin / @jasonzeng124 |

¹ Mixed provider (seeds 0-2 Modal, 3-9 RunPod), grandfathered; 004's RunPod
seeds were significantly slower than its Modal seeds (1088 vs 655 s), see its
README. Ranked records now run all seeds on one provider.

Record 004 is the same recipe as 003 with the guardrail measured; 005 is the
same RL recipe on a 2x faster trainer (shared-prefix KV reuse, compiled
static-cache decode, full-softmax sampler). Both were first submitted with 3
seeds (004: 655 ± 116 s, 005: 511 ± 123 s — a difference that was *not*
significant, p = 0.11) and then extended to 10 seeds each to measure the real
variance: 004's first three seeds turned out to be lucky draws, and the
std of time-to-threshold is 200-350 s, i.e. 40 % of the mean. Reducing that
variance is itself a good record: per
[docs/threshold_calibration.md](docs/threshold_calibration.md) 97 % of it is
steps-to-threshold (learning speed), not host speed or eval noise. Negative
results (dynamic sampling, T=1.0, Dr. GRPO advantages) are written up in
record 005's README.

Not ranked: [000_smoke_cpu](records/track_a/000_smoke_cpu),
[001_smoke_gpu_qwen05b](records/track_a/001_smoke_gpu_qwen05b) (pipeline
smoke tests), and the RTX 4090 reproduction of record 002 in
[002_curriculum_qwen05b_h100/repro_4090](records/track_a/002_curriculum_qwen05b_h100/repro_4090).


## Credits

Extends the modded-nanogpt / nanochat lineage. Every accepted record lists its author in the
leaderboard.
