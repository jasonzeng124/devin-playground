#!/usr/bin/env bash
# Per-item runner for a self-running AWS box (scripts/aws/box_boot.sh sets RUNNER to this): runs the real runner
# and, when it fails with saved state left behind, runs it again in place so the item resumes from that state
# instead of being recorded as FAILED. Nobody is there to requeue it by hand.
#
#   bash scripts/aws/box_runner.sh <config dir> <seed>     (called by scripts/slurm/queue_worker.sbatch)
#   RUNNER_INNER  the real runner (default scripts/research/run_item.sh)
#   BOX_RETRIES   attempts in a row that may fail without the saved state moving on (default 3)
set -uo pipefail
dir=${1:?config dir} seed=${2:?seed}
inner=${RUNNER_INNER:-scripts/research/run_item.sh}
state="${CKPT_DIR:?}/$dir/resume/$seed/state.pt"
running="queue/running/${SLURM_JOB_ID:-none}"
stuck=0 last=
while true; do
  bash "$inner" "$dir" "$seed" && exit 0
  rc=$?
  # no state: nothing to resume from. An empty running file: the inner runner has already put the item back.
  [[ -s $running && -e $state ]] || exit "$rc"
  now=$(stat -c %Y "$state")
  if [[ $now == "$last" ]]; then stuck=$(( stuck + 1 )); else stuck=1; fi
  last=$now
  (( stuck >= ${BOX_RETRIES:-3} )) && exit "$rc"
  echo "box_runner: $dir $seed failed (exit $rc); resuming from its saved state in 30 s (try $stuck without progress)" >&2
  sleep 30
done
