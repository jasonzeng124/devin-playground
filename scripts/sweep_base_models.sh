#!/usr/bin/env bash
set -euo pipefail

mkdir -p results
python -m rlvr_speedrun.eval \
  --models "HuggingFaceTB/SmolLM2-135M,HuggingFaceTB/SmolLM2-360M,Qwen/Qwen2.5-0.5B,HuggingFaceTB/SmolLM2-1.7B,Qwen/Qwen2.5-1.5B" \
  --limit 500 \
  --few-shot 3 \
  --batch-size 64 \
  --max-new-tokens 64 \
  --device auto \
  --out results/base_sweep.json
