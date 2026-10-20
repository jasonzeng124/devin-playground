#!/bin/bash
# Work-queue worker: pops "<record dir> <seed>" lines from ~/rlvr/queue/$QUEUE.txt (QUEUE=a:b: from a.txt
# while it has lines, then b.txt) and runs each
# with scripts/run_seed.sh until the queue is empty (or the job's time is nearly up). Submit as
# many workers as the per-user GPU limits allow; they share the queue under flock.
#
#   sbatch -p mit_preemptable --time=2-00:00:00 scripts/slurm/queue_worker.sbatch
#   sbatch -p mit_normal_gpu --time=06:00:00 scripts/slurm/queue_worker.sbatch
#   sbatch -p mit_normal_gpu --time=06:00:00 --gres=gpu:h100:1 --exclude=node1708,node1709,node4508 \
#     --export=ALL,QUEUE=record,REQUIRE_GPU="NVIDIA H100 80GB HBM3" scripts/slurm/queue_worker.sbatch
#
# QUEUE (default "pending") picks the queue file; keep ranked-record seeds, which must run on the
# reference H100, in their own queue served only by --gres=gpu:h100:1 workers. gpu:h100 also
# matches Engaging's "NVIDIA H100 NVL" cards (the 2-GPU nodes node1708, node1709, node4508), so
# record workers also need --exclude=node1708,node1709,node4508 and REQUIRE_GPU="NVIDIA H100 80GB HBM3"
# (a worker on any other GPU exits before popping anything).
#
# A preempted (requeued) worker first redoes the item it was running; the partial seed is
# filed under <record>/extra/ as with calibrate_seed.sbatch. Outside the record queue, training
# continues from saved state (run_seed.sh: CKPT_DIR, RESUME_EVERY_S). Fill the queue with
# scripts/slurm/make_sweep.py. Finished items are appended to queue/done.txt. RUNNER picks the per-seed
# script (default scripts/run_seed.sh; benchmark v3 records use scripts/run_seed_v3.sh).
#SBATCH --job-name=rlvr-worker
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=%x-%j.out
set -uo pipefail
cd "${RLVR_DIR:-$HOME/rlvr}"  # RLVR_DIR: another checkout (e.g. for screens on a branch)
source "${VENV:-$HOME/rlvr-venv}/bin/activate"  # VENV=~/orcd/pool/vllm-venv for rollout_backend=vllm runs
if [[ -n "${VENV:-}" ]]; then  # vLLM (cu130 wheels) JIT-compiles kernels, so it needs nvcc
  module load cuda/13.0.1
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi
export PYTHON=python DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}  # models are pre-downloaded on the login node; the Hub API times out
export CKPT_DIR=${CKPT_DIR:-$HOME/orcd/pool/rlvr-ckpts}  # checkpoints (~1 GB each) go to the 1 TB pool, not ~/rlvr
Q="${RLVR_DIR:-$HOME/rlvr}/queue"
QUEUE=${QUEUE:-pending}  # one queue name, or several separated by colons: each item comes from the first non-empty one
RESERVE_MIN=${RESERVE_MIN:-90}  # do not start an item with less than this much job time left
mkdir -p "$Q/running"
log() { echo "$(date -u +%FT%TZ) worker $SLURM_JOB_ID ($QUEUE): $*" | tee -a "$Q/worker.log"; }
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ -n "${REQUIRE_GPU:-}" && "$gpu" != "$REQUIRE_GPU" ]]; then
  log "got $gpu on $(hostname), need $REQUIRE_GPU; exiting"
  [[ -s "$Q/running/$SLURM_JOB_ID" ]] && flock "$Q/lock" bash -c 'cat "$0" - <"$1" >"$1.new" && mv "$1.new" "$1"' "$Q/running/$SLURM_JOB_ID" "$Q/${QUEUE%%:*}.txt"
  rm -f "$Q/running/$SLURM_JOB_ID"
  exit 1
fi

minutes_left() {  # squeue %L is [D-]HH:MM:SS, MM:SS or UNLIMITED
  local l d=0 h=0 m=0; l=$(squeue -h -j "$SLURM_JOB_ID" -o %L)
  [[ "$l" == UNLIMITED ]] && { echo 99999; return; }
  [[ "$l" == *-* ]] && { d=${l%%-*}; l=${l#*-}; }
  IFS=: read -ra p <<<"$l"
  case ${#p[@]} in 3) h=${p[0]} m=${p[1]} ;; 2) m=${p[0]} ;; esac
  echo $(( 10#$d * 1440 + 10#$h * 60 + 10#$m ))
}
pop() {
  local q item
  for q in ${QUEUE//:/ }; do
    [[ -s "$Q/$q.txt" ]] || continue
    item=$(flock "$Q/lock" bash -c 'head -1 "$0"; sed -i 1d "$0"' "$Q/$q.txt")
    [[ -n "$item" ]] && { echo "$item"; return; }
  done
}

while true; do
  item=$(cat "$Q/running/$SLURM_JOB_ID" 2>/dev/null)
  if [[ -z "$item" ]]; then
    (( $(minutes_left) < RESERVE_MIN )) && { log "less than $RESERVE_MIN min left; exiting"; break; }
    item=$(pop)
    [[ -z "$item" ]] && { log "queue empty; exiting"; break; }
    echo "$item" >"$Q/running/$SLURM_JOB_ID"
  fi
  read -r record seed <<<"$item"
  if [[ -f "$record/seeds/$seed/train_log.jsonl" || -f "$record/seeds/$seed/pretrain_log.jsonl" ]]; then  # pretrain_log alone: stopped between track D stages
    n=$(ls -d "$record/extra/interrupted_${seed}_"* 2>/dev/null | wc -l)
    mkdir -p "$record/extra" && mv "$record/seeds/$seed" "$record/extra/interrupted_${seed}_$((n + 1))"
    log "restarting $record $seed; partial run kept under extra/"
  fi
  log "start $record $seed on $(hostname) ($gpu)"
  if bash "${RUNNER:-scripts/run_seed.sh}" "$record" "$seed"; then result=ok; else result=FAILED; fi
  log "$result $record $seed"
  echo "$record $seed $result" >>"$Q/done.txt"
  rm -f "$Q/running/$SLURM_JOB_ID"
done
