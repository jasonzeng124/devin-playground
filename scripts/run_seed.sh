#!/usr/bin/env bash
# Train, probe and confirm one seed of a record on the current machine (any provider).
#
#   bash scripts/run_seed.sh records/track_a/<NNN>_<slug> <SEED> [extra grpo flags...]
#
# Writes <record>/seeds/<SEED>/{config.json,train_log.jsonl,eval_log.jsonl,result.json,
# probe.json,confirm.json,stdout.log}. Only the training run is timed; the probe and the
# confirmation eval run afterwards on the saved final/ checkpoint, which is then deleted
# (set KEEP_FINAL=1 to keep it in place, or CKPT_DIR=<dir> to move final/ and any ckpt_<step>/
# from save_every to <dir>/<record>/seeds/<SEED>/, e.g. bulk storage outside the repo). With
# CKPT_DIR and RESUME_EVERY_S=<seconds> set, training saves resumable state to
# <dir>/<record>/resume/<SEED>/ and a rerun (e.g. after preemption) continues from it; the state
# is deleted once the seed finishes. Unranked pilots only: a resumed run is not one timed run. DEVICE defaults to cuda; PYTHON to `uv run python` if uv
# is installed, else `python`.
set -euo pipefail

record=${1:?usage: run_seed.sh <record dir> <seed> [extra grpo flags...]}
seed=${2:?usage: run_seed.sh <record dir> <seed> [extra grpo flags...]}
shift 2
device=${DEVICE:-cuda}
if [[ -z "${PYTHON:-}" ]]; then
  if command -v uv >/dev/null; then PYTHON="uv run python"; else PYTHON=python; fi
fi
out="$record/seeds/$seed"
model=$($PYTHON -c "import json,sys; print(json.load(open(sys.argv[1]))['model'])" "$record/config.json")
mkdir -p "$out"
resume=()
if [[ -n "${CKPT_DIR:-}" && "${QUEUE:-}" != record && "${RESUME_EVERY_S:-1200}" != 0 ]]; then
  resume_dir="$CKPT_DIR/$record/resume/$seed"
  resume=(--resume-dir "$resume_dir" --resume-every-s "${RESUME_EVERY_S:-1200}")
fi

{
  echo "=== train (timed) seed $seed on $(hostname) $(date -u +%FT%TZ)"
  $PYTHON -m rlvr_speedrun.grpo --config "$record/config.json" --device "$device" --seed "$seed" --out-dir "$out" ${resume[@]+"${resume[@]}"} "$@"
  echo "=== capability probe (untimed)"
  $PYTHON -m rlvr_speedrun.capability --model "$out/final" --base "$model" --device "$device" --out "$out/probe.json"
  echo "=== confirmation eval (untimed)"
  $PYTHON -m rlvr_speedrun.confirm --model "$out/final" --device "$device" --out "$out/confirm.json" ${CONFIRM_LIMIT:+--limit "$CONFIRM_LIMIT"}
} 2>&1 | tee "$out/stdout.log"

if [[ -n "${CKPT_DIR:-}" ]]; then
  dest="$CKPT_DIR/$record/seeds/$seed"
  mkdir -p "$dest"
  for ckpt in "$out"/final "$out"/ckpt_*; do
    if [[ -d "$ckpt" ]]; then rm -rf "${dest:?}/$(basename "$ckpt")" && mv "$ckpt" "$dest/"; fi
  done
  if [[ -n "${resume_dir:-}" ]]; then rm -rf "$resume_dir"; fi
elif [[ "${KEEP_FINAL:-0}" != 1 ]]; then
  rm -rf "$out/final"
fi
