#!/bin/bash
# U25 (LOG 2026-11-21 17:59): one rung of the pretraining-size search, outside the rules and unranked. It pretrains
# D 007's architecture (scripts/pretrain_scratch.py: Muon lr 4e-3, the fused loss) at the given shape on the
# two-phase data, then probes the base weights without RL (scripts/research/probe_reasoning.py).
#   first half:  mix_0.2.bin (FineWeb-Edu with 20 % dm_math), D 007's data;
#   second half: mix_0.2.bin 0.3125, dm_math.bin 0.1875, finemath_kw.bin 0.5 by rows, which is FineWeb-Edu 0.25,
#                dm_math 0.25 and FineMath-4plus 0.5 (keyword-filtered: the probe is Countdown, LOG 2026-11-21).
# The three token files are copied to node-local disk first (random reads from the network filesystems are slow).
#
# Usage, from the checkout on a GPU node with the venv active:
#   bash scripts/research/u25_rung.sh <run dir> <width> <layers> <max tokens> <minutes> [seed]
# Weights go to $CKPT_DIR/<run dir>/seeds/<seed>/pretrained and the probe to <run dir>/seeds/<seed>/probe; the
# pretraining log is copied next to the probe.
set -euo pipefail
run=${1:?run dir} width=${2:?width} layers=${3:?layers} max_tokens=${4:?max tokens} minutes=${5:?minutes} seed=${6:-0}
PYTHON=${PYTHON:-python}
CKPT_DIR=${CKPT_DIR:-$HOME/orcd/scratch/rlvr-ckpts}
DATA=${DATA:-$HOME/orcd/pool/data}
FINEMATH=${FINEMATH:-$HOME/orcd/scratch/rlvr-data/r116/finemath_kw.bin}
MICRO=${MICRO:-64}

out=$CKPT_DIR/$run/seeds/$seed/pretrained
here=$run/seeds/$seed
if [[ -e $out/pretrain_log.jsonl ]]; then
  echo "$out/pretrain_log.jsonl exists; not overwriting" >&2
  exit 2
fi
mkdir -p "$here"

local_dir=${LOCAL_DIR:-/tmp}/u25_${SLURM_JOB_ID:-$$}
mkdir -p "$local_dir"
trap 'rm -rf "$local_dir"' EXIT
for f in "$DATA/dm_math/mix_0.2.bin" "$DATA/dm_math/dm_math.bin" "$FINEMATH"; do
  cp "$f" "$local_dir/"
done
echo "=== u25 data copied $(date -u +%FT%TZ): $(ls -l "$local_dir" | tail -n +2 | awk '{print $9, $5}' | tr '\n' ' ')"

mix=$local_dir/mix_0.2.bin
"$PYTHON" scripts/pretrain_scratch.py train --tokens "$mix" --tokenizer "$DATA/tinyllama-tokenizer" --out "$out" \
  --width "$width" --layers "$layers" --micro-batch "$MICRO" --fused-loss --optimizer muon --lr 4e-3 \
  --max-tokens "$max_tokens" --minutes "$minutes" --seed "$seed" \
  --phase2 "$mix:0.3125,$local_dir/dm_math.bin:0.1875,$local_dir/$(basename "$FINEMATH"):0.5" --phase2-start 0.5
cp "$out/pretrain_log.jsonl" "$here/"
rm -rf "$local_dir"
echo "=== u25 pretraining done $(date -u +%FT%TZ); probing"
"$PYTHON" scripts/research/probe_reasoning.py --model "$out" --out-dir "$here/probe"
echo "=== u25 probe done $(date -u +%FT%TZ)"
