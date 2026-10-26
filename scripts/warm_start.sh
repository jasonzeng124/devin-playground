#!/usr/bin/env bash
# Untimed warm start before a seed runs on a node: read the Python environment's large shared libraries
# (torch, the CUDA libraries, vLLM, Triton) and the config's base model into the node's page cache, then
# import the trainer once in a process that sees no GPU (CUDA_VISIBLE_DEVICES=). Nothing here touches the
# GPU. On a cold node the timed stages otherwise read all this over the network filesystems: D 003's
# start-ups took ~16 min a seed on the H200 nodes against 2-3.5 min on the H100, which put its seeds
# over budget (LOG 2026-10-26 07:05). Prints the time it took; never fails the caller.
#
#   bash scripts/warm_start.sh <config.json>         # LIST=1 only prints the files it would read
set -uo pipefail
config=${1:?usage: warm_start.sh <config.json>}
PYTHON=${PYTHON:-python}
t0=$(date +%s)
sp=$($PYTHON -c 'import site; print(site.getsitepackages()[0])')
# bounded globs, no directory walks: the environment lives on ~/orcd/pool
files=("$sp"/torch/lib/*.so* "$sp"/nvidia/*/lib/*.so* "$sp"/vllm/*.so "$sp"/vllm/*/*.so "$sp"/triton/_C/*.so)
model=""
[[ -f $config ]] && model=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1])).get('model', ''))" "$config" 2>/dev/null)
if [[ -n $model && ! -d $model ]]; then  # a hub id: its local snapshot (the runs are offline)
  model=$($PYTHON -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], local_files_only=True))" "$model" 2>/dev/null)
fi
[[ -n $model && -d $model ]] && files+=("$model"/*.safetensors)
if [[ ${LIST:-0} == 1 ]]; then
  ls -lL "${files[@]}" 2>/dev/null | awk '{s += $5; n++} END {printf "%d files, %.1f GB\n", n, s / 1e9}'
  exit 0
fi
cat "${files[@]}" >/dev/null 2>&1
CUDA_VISIBLE_DEVICES= $PYTHON -c 'import torch, transformers, vllm, rlvr_speedrun.grpo' >/dev/null 2>&1
echo "warm start: $(( $(date +%s) - t0 )) s"
exit 0
