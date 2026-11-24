#!/bin/bash
# U25 (LOG 2026-11-24): memory smoke of the 6x rung's shape (1280 x 32) before the 11-25 box (8x H100, 80 GB).
# Under torchrun with every GPU on the node and --ddp, as on the box, it pretrains a few steps at each micro-batch
# given, with each process's memory capped at <cap GiB> (scripts/research/mem_cap_run.py), and
# --resume-every-minutes 1 so the resume save runs too. Each micro-batch's stdout (with the `mem_cap_run` peak
# lines) goes to <dir>/m<micro>.log and its pretrain_log.jsonl to <dir>/m<micro>_pretrain_log.jsonl; the weights
# stay on node-local disk and are deleted.
#
# Usage, from the checkout on a GPU node with the venv active:
#   bash scripts/research/u25_mem_smoke.sh <dir> [cap GiB, default 78] [micro-batches, default "64 32"]
set -uo pipefail
dir=${1:?dir} cap=${2:-78} micros=${3:-64 32}
PYTHON=${PYTHON:-python}
DATA=${DATA:-$HOME/orcd/pool/data}
G=$(nvidia-smi -L | wc -l)
local_dir=${LOCAL_DIR:-/tmp}/u25mem_${SLURM_JOB_ID:-$$}
mkdir -p "$local_dir" "$dir"
trap 'rm -rf "$local_dir"' EXIT
cp "$DATA/dm_math/mix_0.2.bin" "$local_dir/" || exit 3
echo "=== u25 memory smoke on $(hostname), $G GPUs, cap $cap GiB, $(date -u +%FT%TZ)"
for m in $micros; do
  echo "=== micro-batch $m $(date -u +%FT%TZ)"
  timeout 1200 "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$G" scripts/research/mem_cap_run.py "$cap" \
    scripts/pretrain_scratch.py train --tokens "$local_dir/mix_0.2.bin" --tokenizer "$DATA/tinyllama-tokenizer" \
    --width 1280 --layers 32 --micro-batch "$m" --fused-loss --optimizer muon --lr 4e-3 --max-tokens 10485760 \
    --minutes 6 --resume-every-minutes 1 --seed 0 --ddp --out "$local_dir/m$m" > "$dir/m$m.log" 2>&1
  echo "rc=$? $(grep -h '^mem_cap_run' "$dir/m$m.log" | tr '\n' ' ')"
  cp "$local_dir/m$m/pretrain_log.jsonl" "$dir/m${m}_pretrain_log.jsonl" 2>/dev/null
  rm -rf "$local_dir/m$m"
done
echo "=== u25 memory smoke done $(date -u +%FT%TZ)"
