#!/usr/bin/env bash
# Box runner for one U25 rung (unranked, outside the rules; LOG 2026-11-24). A plan line's queue worker on GPU 0
# calls it through scripts/aws/box_runner.sh (RUNNER_INNER=scripts/aws/u25_box_item.sh) with the item
# "<run dir> <seed>". The rung's shape is in <run dir>/config.json under "u25" (width, layers, max_tokens, minutes,
# micro). It runs scripts/research/u25_rung.sh on every GPU in U25_GPUS with DDP, saving a resume state every
# U25_RESUME_MIN minutes, and runs it again (from that state) when it fails, up to U25_TRIES times; while no state
# exists yet, an out-of-memory failure halves the micro-batch first (64 -> 32 -> 16; the step stays 524,288 tokens).
# When it ends, either way, it:
#   - copies the trained weights to U25_S3 (if set and they exist), since the box's disk goes when the box does;
#   - hands the GPUs to the screens: under queue/lock it appends the queue file U25_AFTER to the queue file U25_TO
#     and empties U25_AFTER (both must be set; a second call does nothing).
# It always exits 0, so the box's keeper never puts the item back onto GPUs that the screens are using by then;
# whether the rung trained is in <run dir>/seeds/<seed>/stdout.log and the pretraining log.
#   U25_GPUS=0,1,2,3,4,5,6,7 U25_AFTER=queue/box_1125_after.txt U25_TO=queue/box_1125.txt \
#     bash scripts/aws/u25_box_item.sh calibration/research_r130/u25_6x 0
set -uo pipefail
dir=${1:?run dir} seed=${2:?seed}
PYTHON=${PYTHON:-python}
here=$dir/seeds/$seed
mkdir -p "$here"
log=$here/stdout.log

release() {
  [[ -n ${U25_AFTER:-} && -n ${U25_TO:-} ]] || return 0
  flock queue/lock bash -c '[[ -s $0 ]] || exit 0; cat "$0" >>"$1" && : >"$0"' "$U25_AFTER" "$U25_TO"
  echo "=== u25_box_item: $U25_AFTER handed to $U25_TO $(date -u +%FT%TZ)" >>"$log"
}
trap release EXIT

read -r width layers max_tokens minutes micro < <("$PYTHON" -c '
import json, sys
u = json.load(open(sys.argv[1]))["u25"]
print(u["width"], u["layers"], u["max_tokens"], u["minutes"], u.get("micro", 64))' "$dir/config.json") || {
  echo "=== u25_box_item: no u25 section in $dir/config.json" >>"$log"; exit 0; }
gpus=${U25_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
export CUDA_VISIBLE_DEVICES=$gpus
nproc=$(awk -F, '{print NF}' <<<"$gpus")
out=${CKPT_DIR:?}/$dir/seeds/$seed/pretrained
state=$out/resume/state.pt

try=1
while (( try <= ${U25_TRIES:-4} )); do
  echo "=== u25_box_item: try $try on GPUs $gpus, micro $micro $(date -u +%FT%TZ)" >>"$log"
  NPROC=$nproc RESUME_MIN=${U25_RESUME_MIN:-30} MICRO=$micro LOCAL_DIR=${U25_LOCAL_DIR:-none} PYTHON=$PYTHON \
    bash scripts/research/u25_rung.sh "$dir" "$width" "$layers" "$max_tokens" "$minutes" "$seed" >>"$log" 2>&1
  rc=$?
  (( rc == 0 )) && break
  echo "=== u25_box_item: try $try failed (exit $rc) $(date -u +%FT%TZ)" >>"$log"
  if [[ ! -e $state ]] && (( micro > 16 )) && tail -n 200 "$log" | grep -qi "out of memory"; then
    micro=$(( micro / 2 ))
    echo "=== u25_box_item: out of memory before any saved state; micro-batch $micro" >>"$log"
  fi
  try=$(( try + 1 ))
  sleep "${U25_RETRY_S:-30}"
done
if [[ -n ${U25_S3:-} && -e $out/model.safetensors ]]; then
  aws s3 cp --only-show-errors --recursive "$out/" "${U25_S3%/}/$dir/seeds/$seed/pretrained/" --exclude 'resume/*' \
    </dev/null >>"$log" 2>&1 && echo "=== u25_box_item: weights copied to S3 $(date -u +%FT%TZ)" >>"$log"
fi
exit 0
