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
# Options (all off by default, so the 3x rung's call behaves as before):
#   NPROC=N       N > 1: torchrun with N processes on this node and --ddp (pretrain_scratch.py's DDP path).
#   RESUME_MIN=M  M > 0: save a resume state every M minutes and continue from it if one exists, so the rung
#                 can be rerun after a crash or a box restart. Then a finished run (model.safetensors present)
#                 skips straight to the probe, a log without a state is moved aside to pretrain_log.failed.N.jsonl
#                 and training starts again, and the resume state is deleted after training ends.
#   LOCAL_DIR=none  read the token files in place instead of copying them (a box's disk is already local).
set -euo pipefail
run=${1:?run dir} width=${2:?width} layers=${3:?layers} max_tokens=${4:?max tokens} minutes=${5:?minutes} seed=${6:-0}
PYTHON=${PYTHON:-python}
CKPT_DIR=${CKPT_DIR:-$HOME/orcd/scratch/rlvr-ckpts}
DATA=${DATA:-$HOME/orcd/pool/data}
FINEMATH=${FINEMATH:-$HOME/orcd/scratch/rlvr-data/r116/finemath_kw.bin}
MICRO=${MICRO:-64}
NPROC=${NPROC:-1}
RESUME_MIN=${RESUME_MIN:-0}

out=$CKPT_DIR/$run/seeds/$seed/pretrained
here=$run/seeds/$seed
train=1 resume_args=()
if (( RESUME_MIN > 0 )); then
  resume_args=(--resume-every-minutes "$RESUME_MIN" --resume)
  if [[ -e $out/model.safetensors ]]; then
    echo "=== u25 $out is already trained; probing only"
    train=0
  elif [[ -e $out/pretrain_log.jsonl && ! -e $out/resume/state.pt ]]; then
    n=0
    while [[ -e $out/pretrain_log.failed.$n.jsonl ]]; do n=$(( n + 1 )); done
    mv "$out/pretrain_log.jsonl" "$out/pretrain_log.failed.$n.jsonl"
    echo "=== u25 a log without a resume state was moved to pretrain_log.failed.$n.jsonl; starting again"
  fi
elif [[ -e $out/pretrain_log.jsonl ]]; then
  echo "$out/pretrain_log.jsonl exists; not overwriting" >&2
  exit 2
fi
mkdir -p "$here"

if (( train )); then
  if [[ ${LOCAL_DIR:-} == none ]]; then
    mix=$DATA/dm_math/mix_0.2.bin dm=$DATA/dm_math/dm_math.bin fm=$FINEMATH
  else
    local_dir=${LOCAL_DIR:-/tmp}/u25_${SLURM_JOB_ID:-$$}
    mkdir -p "$local_dir"
    trap 'rm -rf "$local_dir"' EXIT
    for f in "$DATA/dm_math/mix_0.2.bin" "$DATA/dm_math/dm_math.bin" "$FINEMATH"; do
      cp "$f" "$local_dir/"
    done
    echo "=== u25 data copied $(date -u +%FT%TZ): $(ls -l "$local_dir" | tail -n +2 | awk '{print $9, $5}' | tr '\n' ' ')"
    mix=$local_dir/mix_0.2.bin dm=$local_dir/dm_math.bin fm=$local_dir/$(basename "$FINEMATH")
  fi
  launch=("$PYTHON") ddp_args=()
  if (( NPROC > 1 )); then
    launch=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NPROC") ddp_args=(--ddp)
  fi
  "${launch[@]}" scripts/pretrain_scratch.py train --tokens "$mix" --tokenizer "$DATA/tinyllama-tokenizer" --out "$out" \
    --width "$width" --layers "$layers" --micro-batch "$MICRO" --fused-loss --optimizer muon --lr 4e-3 \
    --max-tokens "$max_tokens" --minutes "$minutes" --seed "$seed" "${ddp_args[@]}" "${resume_args[@]}" \
    --phase2 "$mix:0.3125,$dm:0.1875,$fm:0.5" --phase2-start 0.5
  if [[ -n ${local_dir:-} ]]; then rm -rf "$local_dir"; fi
  if (( RESUME_MIN > 0 )); then rm -rf "$out/resume"; fi
fi
cp "$out/pretrain_log.jsonl" "$here/"
echo "=== u25 pretraining done $(date -u +%FT%TZ); probing"
"$PYTHON" scripts/research/probe_reasoning.py --model "$out" --out-dir "$here/probe"
echo "=== u25 probe done $(date -u +%FT%TZ)"
