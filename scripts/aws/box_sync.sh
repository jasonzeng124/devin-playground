#!/usr/bin/env bash
# Copy what box screens need from Engaging to an AWS capacity-block box, or bring their results back.
# Run from Engaging (the box's security group allows ssh from 18.13.54.0/24); the box mirrors the Engaging
# home layout, so configs' /home/zjason/... paths work there.
#
#   BOX=zjason@100.29.118.132 bash scripts/aws/box_sync.sh to b200 [pool paths ...]
#     the checkout's code, data and every config (no logs, no .git), queue/<q>.txt for each colon-separated
#     queue name, the base model from ~/.cache/huggingface, the TinyLlama tokenizer, and any extra paths
#     under ~ (e.g. orcd/pool/rlvr-ckpts/<arm>/seeds/1/final for a probe). Big: run it as a mit_normal CPU job.
#   BOX=zjason@100.29.118.132 bash scripts/aws/box_sync.sh from box1 calibration/research_r15 [...]
#     finished seeds into the checkout (<arm>/seeds/<i>/), running seeds' logs into $CKPT_DIR (so
#     summarize_screens.py reads them as on Engaging), and the box's worker log, done list and worker
#     outputs as queue/<tag>_worker.log, queue/<tag>_done.txt and slurm_logs/<tag>/. Small: logs only.
set -euo pipefail
cd "$(dirname "$0")/../.."
BOX=${BOX:?set BOX=zjason@<box ip>}
export RSYNC_RSH="ssh -i $HOME/.ssh/rlvr_aws -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=20"
CKPT=${CKPT_DIR:-$HOME/orcd/pool/rlvr-ckpts}
mode=${1:?to or from}
shift
case $mode in
  to)
    queues=${1:?queue names}
    shift
    list=$(mktemp)
    {
      git ls-files rlvr_speedrun scripts tests data pyproject.toml setup.py uv.lock
      find calibration records -name config.json -not -path '*/seeds/*' -not -path '*/extra/*'
      find calibration -name eval.json -not -path '*/seeds/*'
      for q in ${queues//:/ }; do echo "queue/$q.txt"; done
    } >"$list"
    rsync -a --files-from="$list" ./ "$BOX:rlvr-screens/"
    rm -f "$list"
    rsync -aR ~/./.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base ~/./orcd/pool/data/tinyllama-tokenizer "$@" "$BOX:"
    ;;
  from)
    tag=${1:?tag for the box logs}
    shift
    for dir in "$@"; do  # e.g. calibration/research_r15
      rsync -a --include='*/' --include='*.json' --include='*.jsonl' --include='*.log' --exclude='*' \
        "$BOX:rlvr-screens/$dir/" "$dir/"
      mkdir -p "$CKPT/$dir"
      rsync -a --include='*/' --include='*.json' --include='*.jsonl' --exclude='*' "$BOX:orcd/pool/rlvr-ckpts/$dir/" "$CKPT/$dir/"
    done
    mkdir -p "slurm_logs/$tag"
    rsync -a "$BOX:rlvr-screens/queue/worker.log" "queue/${tag}_worker.log" || true
    rsync -a "$BOX:rlvr-screens/queue/done.txt" "queue/${tag}_done.txt" || true
    rsync -a "$BOX:rlvr-screens/slurm_logs/box-*.out" "slurm_logs/$tag/" || true
    ;;
  *) echo "usage: box_sync.sh to <queues> [paths] | from <tag> <dirs>" >&2; exit 2 ;;
esac
