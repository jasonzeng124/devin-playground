#!/usr/bin/env bash
# Run one dev-ladder screen seed (calibration only, never a record): GRPO with <dir>/config.json,
# no test-ladder scoring. Logs go to <dir>/seeds/<seed>/, weights to $CKPT_DIR/<dir>/seeds/<seed>/final.
# If the config has "screen_pretrained": "<record>", track D RL starts from that record's seed-<seed>
# pretrained model (same pretraining as the record, so only the RL changes are compared).
#
#   QUEUE=w42 RUNNER=scripts/slurm/screen_seed.sh sbatch ... scripts/slurm/queue_worker.sbatch
set -euo pipefail
dir=${1:?usage: screen_seed.sh <dir> <seed>}
seed=${2:?usage: screen_seed.sh <dir> <seed>}
PYTHON=${PYTHON:-python}
out="$dir/seeds/$seed"
weights="${CKPT_DIR:-$HOME/orcd/pool/rlvr-ckpts}/$dir/seeds/$seed"
[[ -e "$out/train_log.jsonl" ]] && { echo "$out already holds a run" >&2; exit 1; }
# a preempted, requeued seed starts over: keep the partial run's logs aside rather than appending to them
stamp=$(date +%s)
[[ -d "$weights" ]] && mv "$weights" "$weights.interrupted.$stamp"
[[ -e "$out/stdout.log" ]] && mv "$out/stdout.log" "$out/stdout.interrupted.$stamp.log"
mkdir -p "$out" "$weights"
pre=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1])).get('screen_pretrained', ''))" "$dir/config.json")
model_flag=()
[[ -n "$pre" ]] && model_flag=(--model "${CKPT_DIR:-$HOME/orcd/pool/rlvr-ckpts}/$pre/seeds/$seed/pretrained")
{
  echo "=== screen $dir seed $seed on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) $(date -u +%FT%TZ)"
  $PYTHON -m rlvr_speedrun.grpo --config "$dir/config.json" --device cuda --seed "$seed" --out-dir "$weights" ${model_flag[@]+"${model_flag[@]}"}
  cp "$weights"/*.jsonl "$out/"
  echo "=== done $(date -u +%FT%TZ)"
} 2>&1 | tee "$out/stdout.log"
