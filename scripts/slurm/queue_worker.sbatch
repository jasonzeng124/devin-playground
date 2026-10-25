#!/bin/bash
# Work-queue worker: pops "<record dir> <seed>" lines from ~/rlvr/queue/$QUEUE.txt (QUEUE=a:b: from a.txt
# while it has lines, then b.txt) and runs each
# with scripts/run_seed.sh until the queue is empty (or the job's time is nearly up). Submit as
# many workers as the per-user GPU limits allow; they share the queue under flock.
#
#   sbatch -p mit_preemptable --time=2-00:00:00 scripts/slurm/queue_worker.sbatch
#   sbatch -p mit_normal_gpu --time=06:00:00 scripts/slurm/queue_worker.sbatch
#   sbatch -p mit_normal_gpu --time=06:00:00 --gres=gpu:h200:1 \
#     --export=ALL,QUEUE=record,REQUIRE_GPU="NVIDIA H200" scripts/slurm/queue_worker.sbatch
#
# QUEUE (default "pending") picks the queue file; keep ranked-record seeds, which must run on the
# reference GPU (v3.1: H200 SXM, "NVIDIA H200"), in their own queue served only by
# --gres=gpu:h200:1 workers with REQUIRE_GPU="NVIDIA H200". gpu:h200 also matches the "NVIDIA H200
# NVL" cards (preemptable 4-GPU nodes node4002, node4009, node4010, node5102, node5105, node5201),
# so preemptable record workers also exclude those (a worker on any other GPU exits before popping
# anything). v3.0 records used --gres=gpu:h100:1, REQUIRE_GPU="NVIDIA H100 80GB HBM3".
#
# A preempted (requeued) worker first redoes the item it was running; the partial seed is
# filed under <record>/extra/ as with calibrate_seed.sbatch. Outside the record queue, training
# continues from saved state (run_seed.sh: CKPT_DIR, RESUME_EVERY_S). Fill the queue with
# scripts/slurm/make_sweep.py. Finished items are appended to queue/done.txt. RUNNER picks the per-seed
# script (default scripts/run_seed.sh; benchmark v3 records use scripts/run_seed_v3.sh).
#
# Bad hosts: triton and vLLM build small C extensions with gcc when a run starts, so a host where
# gcc cannot build one fails every item within a minute (node1800 failed 32 in a row on
# 2026-10-21). The worker checks gcc first: on Ubuntu nodes the cluster's Rocky-built gcc misses the
# multiarch header and library directories, which it then adds; if gcc still fails, the worker
# exits before popping anything. After two failures in a row that each took under 5 minutes, it
# puts the second item back at the front of its queue and exits.
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
src=  # the queue the item in hand came from (unknown for an item redone after preemption: the first)
give_back() {  # put the item in hand back at the front of its queue
  [[ -s "$Q/running/$SLURM_JOB_ID" ]] && flock "$Q/lock" bash -c 'touch "$1"; cat "$0" "$1" >"$1.new" && mv "$1.new" "$1"' "$Q/running/$SLURM_JOB_ID" "$Q/${src:-${QUEUE%%:*}}.txt"
  rm -f "$Q/running/$SLURM_JOB_ID"
}
cc_ok() {  # can gcc build a C extension against the system headers here? (sets cc_err if not)
  local d rc
  d=$(mktemp -d) || return 1
  printf '#include <stdlib.h>\nint f(void) { return abs(-1); }\n' >"$d/t.c"
  gcc "$d/t.c" -O1 -shared -fPIC -o "$d/t.so" 2>"$d/err"; rc=$?
  cc_err=$(head -c 300 "$d/err" | tr '\n' ' ')
  rm -rf "$d"
  return $rc
}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ -n "${REQUIRE_GPU:-}" && "$gpu" != "$REQUIRE_GPU" ]]; then
  log "got $gpu on $(hostname), need $REQUIRE_GPU; exiting"
  give_back
  exit 1
fi
if ! cc_ok && [[ -d /usr/include/x86_64-linux-gnu ]]; then
  export C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
  export CPLUS_INCLUDE_PATH=/usr/include/x86_64-linux-gnu${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}
  export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}
  cc_ok && log "gcc on $(hostname) needs the multiarch include and library paths; added"
fi
if ! cc_ok; then
  log "gcc cannot build a C extension on $(hostname) ($cc_err); exiting"
  give_back
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
pop() {  # sets item (empty when every queue is) and src
  local q
  item= src=
  for q in ${QUEUE//:/ }; do
    [[ -s "$Q/$q.txt" ]] || continue
    item=$(flock "$Q/lock" bash -c 'head -1 "$0"; sed -i 1d "$0"' "$Q/$q.txt")
    [[ -n "$item" ]] && { src=$q; return; }
  done
}

quick_fails=0
while true; do
  item=$(cat "$Q/running/$SLURM_JOB_ID" 2>/dev/null)
  if [[ -z "$item" ]]; then
    (( $(minutes_left) < RESERVE_MIN )) && { log "less than $RESERVE_MIN min left; exiting"; break; }
    pop
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
  t0=$SECONDS
  if bash "${RUNNER:-scripts/run_seed.sh}" "$record" "$seed"; then result=ok; else result=FAILED; fi
  if [[ $result == FAILED ]] && (( SECONDS - t0 < 300 )); then
    if (( ++quick_fails >= 2 )); then
      log "FAILED $record $seed after $(( SECONDS - t0 )) s, the second quick failure in a row on $(hostname); put back, exiting"
      give_back
      exit 1
    fi
  else
    quick_fails=0
  fi
  log "$result $record $seed"
  echo "$record $seed $result" >>"$Q/done.txt"
  rm -f "$Q/running/$SLURM_JOB_ID"
  src=
done
