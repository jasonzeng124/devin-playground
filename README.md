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

### Track A — fixed base model, 1xGPU tier

| # | record | base model | time to 50% (mean ± std, N seeds) | final pass rate | author |
|---|--------|------------|-----------------------------------|-----------------|--------|
| 0 | [000_smoke_cpu](records/track_a/000_smoke_cpu) | SmolLM2-135M | — (pipeline smoke test, CPU) | 0.00 | — |
| 1 | [001_smoke_gpu_qwen05b](records/track_a/001_smoke_gpu_qwen05b) | Qwen2.5-0.5B | — (60-step smoke, 1 seed, RTX 4090) | 0.005 | — |

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

# GRPO
uv run python -m rlvr_speedrun.grpo --model Qwen/Qwen2.5-0.5B \
    --group-size 8 --prompts-per-step 16 --max-steps 300 --eval-every 25 \
    --target-solve-rate 0.5 --out-dir records/track_a/001_grpo_baseline/seeds/0

# validate a record before opening a PR
uv run python scripts/validate_record.py records/track_a/001_grpo_baseline
```

On a fresh GPU box (e.g. a RunPod `runpod/pytorch` container) run
`bash scripts/gpu_setup.sh` instead of `uv sync`.

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

Pass rate lifts off zero around 360M–0.5B; **Qwen2.5-0.5B** is the provisional
Track A base. Solves are sparse enough that early GRPO signal comes mostly from
the malformed penalty (see record 001), so a curriculum or lower sampling
temperature is the obvious first real record.

## Good first records

* Tune `lr`, `kl_coef`, `group_size`, `temperature` on the baseline.
* Curriculum: start with 3-number puzzles / small targets, anneal to the eval distribution.
* Reward shaping from the verifier output (e.g. partial credit for using all numbers).
* Rollout throughput: batched generation, KV-cache reuse, `torch.compile`.
* Replace AdamW with Muon on the policy.

## Credits

Extends the modded-nanogpt / nanochat lineage. Contributor credit follows the
modded-nanogpt convention: every accepted record lists its author in the table
above.
