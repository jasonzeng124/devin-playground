# RLVR Speedrun (Countdown)

A speedrun leaderboard for **reinforcement learning with verifiable rewards** —
the RL analogue of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
The goal: teach a small language model to solve Countdown arithmetic puzzles
(combine 4 numbers with `+ - * /`, each used once, to hit a target) **as fast as
possible in wall-clock time**, scored by a deterministic CPU verifier.

Why Countdown: synthetic (infinite data, no leakage), difficulty-tunable, no
saturation ceiling, and the verifier runs in microseconds. See
[RULES.md](RULES.md) for the full rules, tracks, and submission process.

## Leaderboard

Hardware: **one H100 80GB**. Track A base model: **Qwen/Qwen2.5-0.5B**
(pretrained-only checkpoint, no post-training). Threshold: **10% pass rate**
on the frozen 2,000-puzzle eval, 3 seeds.

### Track A — fixed base model, RL only

| # | record | time to 10% (mean ± std, N=3) | final pass rate | guardrail | author |
|---|--------|------------------------------:|----------------:|-----------|--------|
| 1 | [002_curriculum_qwen05b_h100](records/track_a/002_curriculum_qwen05b_h100) | 783 ± 216 s | 0.104 | not measured | Devin / @jasonzeng124 |
| 2 | [003_nokl_deferred_eval_h100](records/track_a/003_nokl_deferred_eval_h100) | 573 ± 214 s | 0.105 | not measured | Devin / @jasonzeng124 |
| 3 | [004_nokl_guardrail_h100](records/track_a/004_nokl_guardrail_h100) | 655 ± 116 s | 0.110 | pass (Δloss +0.001) | Devin / @jasonzeng124 |
| 4 | [005_prefix_kv_compile_h100](records/track_a/005_prefix_kv_compile_h100) | 511 ± 123 s | 0.110 | pass (Δloss +0.002) | Devin / @jasonzeng124 |

Record 004 is the same recipe as 003 with the guardrail measured; 005 is the
same recipe again on a 2x faster trainer (shared-prefix KV reuse, compiled
static-cache decode). Both are listed separately because RL seed variance
(~±120-200 s) is currently larger than most recipe changes. Reducing that
variance is itself a good record. Negative results (dynamic sampling, T=1.0,
Dr. GRPO advantages) are written up in record 005's README.

Not ranked: [000_smoke_cpu](records/track_a/000_smoke_cpu),
[001_smoke_gpu_qwen05b](records/track_a/001_smoke_gpu_qwen05b) (pipeline
smoke tests), and the RTX 4090 reproduction of record 002 in
[002_curriculum_qwen05b_h100/repro_4090](records/track_a/002_curriculum_qwen05b_h100/repro_4090).

### Track B — full stack (pretrain + RL)

No records yet. `X`/`Y` thresholds will be calibrated from the first baseline run.

## Quickstart

```bash
uv sync                                   # or: pip install -e .
uv run pytest -q                          # verifier, GRPO step, record validator

# frozen eval set is committed; regenerate only with --force
uv run python -m rlvr_speedrun.data build-eval --out data/countdown_eval.jsonl

# zero/few-shot base-model eval (this is how the base checkpoint was chosen)
uv run python -m rlvr_speedrun.eval --model Qwen/Qwen2.5-0.5B --few-shot 3 --limit 500

# sweep several base checkpoints, print a markdown table
bash scripts/sweep_base_models.sh

# GRPO (record 005 recipe; drop --compile/--pad-to-multiple for eager generation)
uv run python -m rlvr_speedrun.grpo --model Qwen/Qwen2.5-0.5B \
    --group-size 16 --prompts-per-step 8 --micro-batch-size 128 --temperature 0.8 \
    --lr 5e-6 --kl-coef 0 --curriculum-steps 200 --max-steps 600 \
    --eval-every 25 --eval-start-step 150 --eval-limit 2000 --eval-batch-size 1000 \
    --target-solve-rate 0.1 --compile --pad-to-multiple 64 --device cuda \
    --no-save --seed 0 --out-dir records/track_a/00N_my_record/seeds/0

# validate a record before opening a PR
uv run python scripts/validate_record.py records/track_a/00N_my_record
```

On a fresh GPU box (e.g. a RunPod `runpod/pytorch` container) run
`bash scripts/gpu_setup.sh` instead of `uv sync`. On Modal (`pip install modal`),
`scripts/modal_run.py` runs all seeds in parallel on H100s and copies logs back
(see record 002 for the exact command).
For Track A submissions, run the capability guardrail probe described in
[RULES.md](RULES.md) after training.

## Layout

```
rlvr_speedrun/
  countdown.py     puzzle generator, prompt format, verifier / reward
  data.py          frozen eval set builder, training puzzle stream
  eval.py          few-shot pass-rate eval + multi-model sweep table
  grpo.py          minimal GRPO loop (plain PyTorch + transformers)
  model_utils.py   model/tokenizer loading, stopping criteria
scripts/
  validate_record.py   checks a records/ entry and prints seed statistics
  sweep_base_models.sh base-model pass-rate sweep
  gpu_setup.sh         one-shot setup on a CUDA container
  modal_run.py         run N seeds in parallel on Modal GPUs (+ capability probe)
  capability.py        (rlvr_speedrun/) FineWeb-Edu loss + MMLU-lite guardrail probe
data/countdown_eval.jsonl   2,000 frozen eval puzzles (seed 20240601)
records/<track>/<NNN>_<slug>/  one folder per record (logs, config, README)
results/                     base-model sweep outputs
```

## Open question: minimum viable base model

RLVR needs a non-zero base pass rate, otherwise every GRPO group has identical
rewards and there is no gradient. `scripts/sweep_base_models.sh` evaluates a
ladder of small open checkpoints; results live in `results/`. Track A's frozen
base is the smallest checkpoint that clears ~2% few-shot.

First sweep (500 eval puzzles, 3-shot, greedy, RTX 4090 — `results/base_sweep.json`):

| model | pass rate | malformed | Wilson 95% CI |
|---|---:|---:|---|
| SmolLM2-135M | 0.000 | 0.012 | [0.000, 0.008] |
| SmolLM2-360M | 0.012 | 0.002 | [0.006, 0.026] |
| Qwen2.5-0.5B | 0.026 | 0.058 | [0.015, 0.044] |
| SmolLM2-1.7B | 0.018 | 0.024 | [0.010, 0.034] |
| Qwen2.5-1.5B | 0.038 | 0.014 | [0.025, 0.059] |

Pass rate lifts off zero around 360M–0.5B; **Qwen2.5-0.5B** is the Track A
base. Solves are sparse enough that plain GRPO has almost no signal (record
001); a 2→3→4-number curriculum on the training stream fixes that (record 002:
2.6% → 10% held-out in ~13 min on one H100).

## Good first records

* Tune `lr`, `kl_coef`, `group_size`, `temperature`, `curriculum_steps` on record 005.
* Trigger the full eval off the in-batch solve rate instead of a fixed schedule
  (each eval costs ~10 s on the clock; `--eval-start-step` is a blunt version).
* Reward shaping from the verifier output (e.g. partial credit for using all numbers).
* Rollout throughput: rollouts are now ~70% of a step. HF `generate` spends most
  of a decode step outside the model (`--compile` only compiles the forward); a
  hand-rolled sampling loop over the static cache, or a vLLM/SGLang rollout
  worker, is the obvious next systems record.
* Fewer optimizer steps to threshold: seeds need 250-450 steps with ±100 s
  spread; anything that tightens that (longer/adaptive curriculum, LR schedule,
  Muon instead of AdamW) beats another 20% of step time.

## Credits

Extends the modded-nanogpt / nanochat lineage. Contributor credit follows the
modded-nanogpt convention: every accepted record lists its author in the table
above.
