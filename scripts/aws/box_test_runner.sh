#!/usr/bin/env bash
# Stand-in for scripts/research/run_item.sh when testing a self-running box on a cheap instance (no GPU, no venv):
# "trains" for TEST_STEPS steps of TEST_STEP_S seconds and keeps the files a screen keeps
# (scripts/slurm/screen_seed.sh): logs under $CKPT_DIR/<dir>/seeds/<seed>/ while running, a resumable state under
# $CKPT_DIR/<dir>/resume/<seed>/, and at the end the logs in <dir>/seeds/<seed>/ with a stdout.log ending "=== done".
#
#   RUNNER_INNER=scripts/aws/box_test_runner.sh   (in the box's plan; scripts/aws/box_runner.sh calls it)
#   TEST_FAIL_AT=<step>   each item fails once at that step (a multiple of 3, so a state exists): tests the resume
#   an arm whose name starts with "bad" fails at once, every time: tests FAILED and the single requeue
#   TEST_NEED="<path under ~>=<sha256> ..."   each item first checks these files, and fails when one is missing or
#                         differs: tests what scripts/aws/box_boot.sh fetched from box/home/ (stage with HOME_EXTRA)
set -euo pipefail
dir=${1:?config dir} seed=${2:?seed}
out="$dir/seeds/$seed" weights="${CKPT_DIR:?}/$dir/seeds/$seed" resume="$CKPT_DIR/$dir/resume/$seed"
[[ -e $out/train_log.jsonl ]] && { echo "$out already has results" >&2; exit 1; }
mkdir -p "$out" "$weights" "$resume"
[[ -e $out/stdout.log ]] && mv "$out/stdout.log" "$out/stdout.interrupted.$(date +%s).log"
instance=$(cat /var/lib/cloud/data/instance-id 2>/dev/null || hostname -s)
{
  echo "=== screen $dir seed $seed on $(hostname) (test stub) $(date -u +%FT%TZ)"
  [[ $(basename "$dir") == bad* ]] && { echo "bad arm: failing"; exit 1; }
  for need in ${TEST_NEED:-}; do
    sum=$(sha256sum "$HOME/${need%%=*}" 2>/dev/null | cut -d' ' -f1) || true
    [[ -n $sum && $sum == "${need#*=}" ]] || { echo "TEST_NEED: ~/${need%%=*} is missing or differs"; exit 1; }
    echo "TEST_NEED ok: ~/${need%%=*} ($(stat -c %s "$HOME/${need%%=*}") bytes)"
  done
  step=0
  : >"$weights/train_log.jsonl"
  if [[ -e $resume/state.pt ]]; then  # as the trainer: the logs restart from the copies saved with the state
    step=$(cat "$resume/state.pt")
    cp "$resume/train_log.jsonl" "$weights/train_log.jsonl"
    echo "resumed from step $step"
  fi
  while (( step < ${TEST_STEPS:-24} )); do
    sleep "${TEST_STEP_S:-10}"
    step=$(( step + 1 ))
    echo "{\"step\": $step, \"instance\": \"$instance\", \"time\": \"$(date -u +%FT%TZ)\"}" >>"$weights/train_log.jsonl"
    if (( step % 3 == 0 )); then  # logs first, then the state, as the trainer does
      cp "$weights/train_log.jsonl" "$resume/train_log.jsonl.tmp" && mv "$resume/train_log.jsonl.tmp" "$resume/train_log.jsonl"
      echo "$step" >"$resume/state.tmp" && mv "$resume/state.tmp" "$resume/state.pt"
    fi
    if [[ ${TEST_FAIL_AT:-0} == "$step" && ! -e $resume/failed_once ]]; then
      touch "$resume/failed_once"
      echo "test failure at step $step"
      exit 1
    fi
  done
  cp "$weights"/*.jsonl "$out/"
  echo "=== done $(date -u +%FT%TZ)"
} 2>&1 | tee "$out/stdout.log"
rm -rf "$resume"
rmdir "${resume%/*}" 2>/dev/null || true
