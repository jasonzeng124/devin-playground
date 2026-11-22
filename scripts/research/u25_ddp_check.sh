#!/bin/bash
# U25 (LOG 2026-11-22): GPU check of scripts/pretrain_scratch.py's --ddp and --resume before the 6x rung on the
# 11-25 box. On one node, with every GPU it has (G), each arm pretrains D 007's shape (768 x 12, Muon lr 4e-3, the
# fused loss) on mix_0.2.bin, copied to node-local disk, for 1,000 steps of 131,072 tokens:
#   single:  one process without --ddp (the existing path);
#   ddp:     torchrun with G processes and --ddp;
#   resume:  as ddp with --resume-every-minutes 0.5, stopped by SIGTERM after <stop> seconds (rc 124 in
#            resume/first_rc.txt), then run again with --resume to the end.
# Each arm's pretrain_log.jsonl and stdout go to <dir>/<arm>/; the weights stay on node-local disk and are deleted.
# scripts/research/u25_ddp_read.py <dir> reads them against the rule.
#
# Usage, from the checkout on a GPU node with the venv active:
#   bash scripts/research/u25_ddp_check.sh <dir> [stop seconds, default 100]
set -uo pipefail
dir=${1:?dir} stop=${2:-100}
PYTHON=${PYTHON:-python}
DATA=${DATA:-$HOME/orcd/pool/data}
G=$(nvidia-smi -L | wc -l)
local_dir=${LOCAL_DIR:-/tmp}/u25ddp_${SLURM_JOB_ID:-$$}
mkdir -p "$local_dir" "$dir"
trap 'rm -rf "$local_dir"' EXIT
cp "$DATA/dm_math/mix_0.2.bin" "$local_dir/" || exit 3
echo "=== u25 ddp check on $(hostname), $G GPUs, data copied $(date -u +%FT%TZ)"

common=(--tokens "$local_dir/mix_0.2.bin" --tokenizer "$DATA/tinyllama-tokenizer" --width 768 --layers 12
        --micro-batch 64 --batch-tokens 131072 --max-tokens 131072000 --minutes 60 --fused-loss --optimizer muon
        --lr 4e-3 --seed 0)
torchrun=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$G")

run_arm() {  # <arm> <command ...>
  local arm=$1
  shift
  mkdir -p "$dir/$arm"
  echo "=== $arm $(date -u +%FT%TZ)"
  "$@" >> "$dir/$arm/stdout.log" 2>&1
  echo "rc=$? $(tail -n 1 "$dir/$arm/stdout.log" | cut -c1-200)"
}

run_arm single "$PYTHON" scripts/pretrain_scratch.py train "${common[@]}" --out "$local_dir/single"
cp "$local_dir/single/pretrain_log.jsonl" "$dir/single/"
run_arm ddp "${torchrun[@]}" scripts/pretrain_scratch.py train "${common[@]}" --ddp --out "$local_dir/ddp"
cp "$local_dir/ddp/pretrain_log.jsonl" "$dir/ddp/"

mkdir -p "$dir/resume"
echo "=== resume, first part (stopped after $stop s) $(date -u +%FT%TZ)"
timeout -s TERM "$stop" "${torchrun[@]}" scripts/pretrain_scratch.py train "${common[@]}" --ddp \
  --resume-every-minutes 0.5 --out "$local_dir/resume" > "$dir/resume/stdout.log" 2>&1
echo $? > "$dir/resume/first_rc.txt"
echo "rc=$(cat "$dir/resume/first_rc.txt") (124: stopped by the timeout); $(ls -l "$local_dir/resume/resume/" | tail -n +2)"
run_arm resume "${torchrun[@]}" scripts/pretrain_scratch.py train "${common[@]}" --ddp --resume-every-minutes 0.5 \
  --resume --out "$local_dir/resume"
cp "$local_dir/resume/pretrain_log.jsonl" "$dir/resume/"
"$PYTHON" scripts/research/u25_ddp_read.py "$dir"
echo "=== u25 ddp check done $(date -u +%FT%TZ)"
