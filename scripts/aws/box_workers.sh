#!/usr/bin/env bash
# Runs ON an AWS capacity-block box (scripts/aws/box_launch.sh), as zjason from ~/rlvr-screens:
# starts PER_GPU queue workers per GPU (scripts/slurm/queue_worker.sbatch outside Slurm), each
# pinned to its GPU with CUDA_VISIBLE_DEVICES, popping from queue/<QUEUE>.txt like the Engaging
# workers. Workers stop starting items when fewer than RESERVE_MIN minutes remain before
# DEADLINE (default: the block end minus 35 min, since AWS may terminate the box 30 min early).
# With several workers per GPU, give each run a matching share of memory (vllm_gpu_memory in the
# configs) and stagger the starts (STAGGER_S), since vLLM sizes its cache from the free memory.
#
#   QUEUE=b200 PER_GPU=2 RUNNER=scripts/slurm/screen_seed.sh bash scripts/aws/box_workers.sh
#   pkill -u zjason -f queue_worker.sbatch    # (on the box only) stop them
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${QUEUE:?set QUEUE}"
PER_GPU=${PER_GPU:-1} STAGGER_S=${STAGGER_S:-90}
export RLVR_DIR=$PWD RUNNER=${RUNNER:-scripts/slurm/screen_seed.sh} QUEUE
export VENV=${VENV:-$HOME/orcd/pool/vllm-venv} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} RESERVE_MIN=${RESERVE_MIN:-60}
export DEADLINE=${DEADLINE:-$(( $(cat /etc/rlvr-box.end) - 35 * 60 ))}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=/usr/local/cuda/bin:$PATH
ngpu=$(nvidia-smi -L | wc -l)
mkdir -p slurm_logs
for k in $(seq 1 "$PER_GPU"); do
  for g in $(seq 0 $((ngpu - 1))); do
    id=box-$(hostname -s)-g$g-$k
    CUDA_VISIBLE_DEVICES=$g SLURM_JOB_ID=$id setsid nohup bash scripts/slurm/queue_worker.sbatch >"slurm_logs/$id.out" 2>&1 &
    echo "started $id"
  done
  (( k < PER_GPU )) && sleep "$STAGGER_S"
done
