#!/bin/bash
# Talk to saved models (scripts/chat.py) on a compute node, never on a login node.
#
#   scripts/chat.sh [model]          e.g. scripts/chat.sh a-best   (then /help inside)
#
# By default this takes CPUs only on mit_quicktest: it starts within minutes, lasts 15 minutes and
# uses none of the six GPU slots. Track D models answer at once; a 1.7B track A model is slower on
# CPUs. For a longer session (the queue can take hours):
#   CHAT_PARTITION=mit_normal CHAT_TIME=2:00:00 scripts/chat.sh a-best
# and for a GPU, which counts against the per-user GPU caps like any other job:
#   CHAT_PARTITION=mit_preemptable CHAT_GRES=gpu:l40s:1 CHAT_TIME=2:00:00 scripts/chat.sh a-best
# Other knobs: CHAT_CPUS (16), CHAT_MEM (32G), CHAT_ARGS (extra chat.py flags), VENV, CKPT_DIR.
set -eu
cd "$(dirname "$0")/.."
VENV=${VENV:-$HOME/orcd/pool/vllm-venv}
model=${1:-}
# the venv's torch needs the compiler module's libstdc++ when one is loaded (as in packed_worker.sbatch)
lib=$("${CXX:-g++}" -print-file-name=libstdc++.so.6 2>/dev/null || true)
[[ $lib == /* ]] && export LD_LIBRARY_PATH=$(dirname "$lib")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
pty=()
[[ -t 0 ]] && pty=(--pty)
export HF_HUB_OFFLINE=1  # exported, not `env ...`: a file named env in ~/.local/bin can shadow the command
exec srun "${pty[@]}" -p "${CHAT_PARTITION:-mit_quicktest}" ${CHAT_GRES:+--gres="$CHAT_GRES"} -c "${CHAT_CPUS:-16}" \
  --mem="${CHAT_MEM:-32G}" -t "${CHAT_TIME:-0:15:00}" --job-name=rlvr-chat \
  "$VENV/bin/python" scripts/chat.py ${model:+--model "$model"} ${CHAT_ARGS:-}
