#!/bin/bash
# One plateau-calibration seed on a Slurm cluster (written for MIT ORCD Engaging).
#
#   sbatch -p mit_normal_gpu --time=06:00:00 --export=ALL,SEED=4 scripts/slurm/calibrate_seed.sbatch
#   sbatch -p mit_preemptable --time=2-00:00:00 --export=ALL,SEED=6 scripts/slurm/calibrate_seed.sbatch
#
# Expects the repo at ~/rlvr and the reference venv at ~/rlvr-venv (torch 2.5.1). A preempted
# job is requeued; the rerun files the partial seed under extra/ and starts it over.
#
# Keep the GPUs busy: with CHAIN_MAX=<n>, each finished job claims the next unused seed number
# (counter in $RECORD/next_seed, under flock) and submits it with the same partition and time,
# until seed n. Seed the counter first, e.g. `echo 14 > calibration/plateau_005/next_seed`.
#SBATCH --job-name=rlvr-plateau
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=%x-%j.out
set -uo pipefail
SEED=${SEED:?pass --export=ALL,SEED=<n>}
RECORD=${RECORD:-calibration/plateau_005}
cd ~/rlvr
source ~/rlvr-venv/bin/activate
export PYTHON=python DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$HOME/.cache/huggingface
status() { echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID seed $SEED: $*" | tee -a "$RECORD/slurm_status.log"; }

if [[ -f "$RECORD/seeds/$SEED/train_log.jsonl" ]]; then
  n=$(ls -d "$RECORD/extra/interrupted_${SEED}_"* 2>/dev/null | wc -l)
  mkdir -p "$RECORD/extra" && mv "$RECORD/seeds/$SEED" "$RECORD/extra/interrupted_${SEED}_$((n + 1))"
  status "restart ${SLURM_RESTART_COUNT:-0}; partial run kept under extra/"
fi
status "start on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
if bash scripts/run_seed.sh "$RECORD" "$SEED" ${EXTRA_FLAGS:-}; then status ok; else status FAILED; fi

if [[ -n "${CHAIN_MAX:-}" ]]; then
  next=$(flock "$RECORD/next_seed" bash -c 'n=$(cat "$0"); (( n <= $1 )) && echo $((n + 1)) > "$0" && echo "$n"' "$RECORD/next_seed" "$CHAIN_MAX")
  if [[ -n "$next" ]]; then
    part=$(squeue -h -j "$SLURM_JOB_ID" -o %P) limit=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
    job=$(sbatch --parsable -D "$HOME/rlvr/slurm_logs" -p "$part" --time="$limit" --export=ALL,SEED="$next",CHAIN_MAX="$CHAIN_MAX" "$HOME/rlvr/scripts/slurm/calibrate_seed.sbatch")
    status "chained seed $next as job $job"
  fi
fi
