#!/bin/bash
# Packed worker: PER_GPU queue workers share this job's one GPU. Each is scripts/slurm/queue_worker.sbatch
# run as on a box (outside Slurm): named <job>-1 .. <job>-PER_GPU, with DEADLINE set to this job's end.
# Only for step-capped screens, whose results don't depend on the GPU's speed or on a neighbour, and
# never for record items (they must run alone on the reference GPU). Each config needs its share of
# memory: r15's vllm_gpu_memory 0.2 and micro_batch_tokens 4096 fit two fp32 track A runs on a B200
# (LOG 2026-10-26 17:45). The workers start STAGGER_S apart, since vLLM sizes its cache from the free
# memory. A requeued job restarts all of them, and each first redoes the item in its hand (screens
# resume from their hourly state on mit_preemptable).
#
#   sbatch -p mit_preemptable --time=2-00:00:00 --time-min=06:30:00 --gres=gpu:b200:1 -c 16 --mem=128G \
#     --export=ALL,RLVR_DIR=$HOME/rlvr-screens,RUNNER=scripts/research/run_item.sh,VENV=$HOME/orcd/pool/vllm-venv,HF_HUB_OFFLINE=1,QUEUE=b200,PER_GPU=2,RESERVE_MIN=420 \
#     scripts/slurm/packed_worker.sbatch
#SBATCH --job-name=rlvr-packed
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --output=%x-%j.out
set -uo pipefail
cd "${RLVR_DIR:-$HOME/rlvr}"
job=$SLURM_JOB_ID n=${PER_GPU:-2}
# The job's end from the time squeue says is left ([D-]HH:MM:SS, as queue_worker.sbatch reads it), not from
# SLURM_JOB_END_TIME, which put two B200 workers with 2 days left under 4 h (LOG 2026-10-28 20:25).
left=$(squeue -h -j "$job" -o %L) days=0 secs=0
[[ $left == UNLIMITED ]] && left=30-00:00:00
[[ $left == *-* ]] && { days=${left%%-*}; left=${left#*-}; }
IFS=: read -ra parts <<<"$left"
for x in "${parts[@]}"; do secs=$(( secs * 60 + 10#$x )); done
export DEADLINE=${DEADLINE:-$(( $(date +%s) + 10#$days * 86400 + secs ))}
# On a B200, vLLM's attention (FlashInfer) JIT-builds kernels with $CXX (Spack GCC 12 in Engaging's default
# environment) that need its libstdc++, newer than the Rocky 8 system one, so it goes first (LOG 2026-10-28).
lib=$("${CXX:-g++}" -print-file-name=libstdc++.so.6 2>/dev/null)
[[ $lib == /* ]] && export LD_LIBRARY_PATH=$(dirname "$lib")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# On some nodes the Lustre client hangs on the venv: workers then sat in D state for an hour or more without
# logging a line (node5200 and node5100, LOG 2026-11-14). Give up on such a node at once, holding no item.
if [[ -n "${VENV:-}" ]] && ! timeout 60 stat "$VENV/bin/activate" >/dev/null 2>&1; then
  echo "$(date -u +%FT%TZ) worker $job (${QUEUE:-}): $VENV/bin/activate unreadable on $(hostname) in 60 s; exiting" \
    | tee -a queue/worker.log
  exit 1
fi
# On node5800-c1 and node5702-c1 (LOG 2026-11-24) ~/orcd/scratch was not mounted: every item failed in seconds
# ("mkdir: cannot create directory '/home/zjason/orcd/scratch': File exists") and was marked FAILED.
if [[ -n "${CKPT_DIR:-}" ]] && ! timeout 60 stat -L "$(dirname "$CKPT_DIR")" >/dev/null 2>&1; then
  echo "$(date -u +%FT%TZ) worker $job (${QUEUE:-}): $(dirname "$CKPT_DIR") unreadable on $(hostname) in 60 s; exiting" \
    | tee -a queue/worker.log
  exit 1
fi
for k in $(seq 1 "$n"); do
  SLURM_JOB_ID=$job-$k bash scripts/slurm/queue_worker.sbatch &
  (( k < n )) && sleep "${STAGGER_S:-120}"
done
wait
