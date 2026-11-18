#!/usr/bin/env bash
# Runs an AWS box with nobody logged in. The instance's user data (scripts/aws/box_userdata.sh, in a launch template
# made by scripts/aws/box_auto.sh) unpacks the staged code and starts this as the box's user; a crontab @reboot line
# starts it again after a reboot. It fetches what the queued runs need, starts the workers of the plan, and keeps
# sending results and saved state to S3 so that a later instance of the same box can carry on where this one stopped.
#
#   bash scripts/aws/box_boot.sh s3://<bucket>/box/<name>        log: ~/box_boot.log   markers: ~/.box_boot/
#
# In S3 (written by scripts/aws/box_s3.sh on Engaging, read here):
#   box/<name>/in/plan           one scripts/aws/box_workers.sh call per line (QUEUE=... GPUS="0 1" PER_GPU=2 ...);
#                                GPUS must be given. A line "boot: NAME=value ..." sets PUSH_S, STATE_S, KEEPER_S,
#                                NEED_VENV or SHUTDOWN_WHEN_IDLE below.
#   box/<name>/in/queue/<q>.txt  the queues
#   box/<name>/in/home.txt       paths under ~ to copy from box/home/ (pretrained checkpoints, tokenizer); a path
#                                ending in .tar is one tar file there, unpacked in place (the base model's cache)
# Written here:
#   box/<name>/out/rlvr-screens/ finished and running seeds' files, queue/snap.tar (scripts/aws/box_queue.py),
#                                queue/worker.log, slurm_logs/
#   box/<name>/out/rlvr-ckpts/   the json and jsonl of running seeds
#   box/<name>/out/status.txt    what the box is doing; out/log/<instance id>/ holds this script's and cloud-init's log
#   box-state/<name>/<config dir>/resume/<seed>/   the saved state of each unfinished seed; removed once it is done
#
# By hand on the box: `touch queue/keeper.off` stops the restarting of dead workers and the requeue of failed items.
set -uo pipefail
P=${1:?usage: box_boot.sh s3://<bucket>/box/<name>}
P=${P%/}
[[ -e /etc/rlvr-box.end ]] || { echo "box_boot: no /etc/rlvr-box.end: this is not a box" >&2; exit 1; }
bucket=${P#s3://} key=${P#s3://*/}
bucket=${bucket%%/*} name=${key##*/}
H=s3://$bucket/box/home S=s3://$bucket/box-state/$name
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH
cd "$(dirname "$0")/../.." || exit 1
export CKPT_DIR=${CKPT_DIR:-$HOME/orcd/pool/rlvr-ckpts}
mark=$HOME/.box_boot
mkdir -p "$mark/states" "$mark/restarts" queue/staged queue/running slurm_logs "$CKPT_DIR"
exec 9>"$mark/lock"  # one box_boot per box; every long-lived child is started with 9>&- so it does not hold this
flock -n 9 || { echo "box_boot: already running"; exit 0; }

log() { echo "$(date -u +%FT%TZ) box_boot: $*"; }
try() {
  local n
  for n in 1 2 3 4 5; do "$@" && return 0; sleep $(( n * 10 )); done
  log "FAILED after 5 tries: $*"
  return 1
}
imds() {  # instance metadata (IMDSv2)
  local token
  token=$(curl -s -f -m 5 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300") || return 1
  curl -s -f -m 5 -H "X-aws-ec2-metadata-token: $token" "http://169.254.169.254/latest/meta-data/$1"
}
list_keys() {  # every key under a prefix, one per line; never takes an S3 error for "nothing there"
  local out
  until out=$(aws s3api list-objects-v2 --bucket "$bucket" --prefix "$1" --query 'Contents[].Key' --output text); do
    log "cannot list s3://$bucket/$1; trying again" >&2
    sleep 20
  done
  tr '\t' '\n' <<<"$out" | grep -vx -e None -e ''
}

region=$(imds placement/region) || region=
export AWS_DEFAULT_REGION=${region:-us-east-1}
iid=$(imds instance-id) || iid=$(cat /var/lib/cloud/data/instance-id 2>/dev/null || hostname -s)
end=$(cat /etc/rlvr-box.end)
host=$(hostname -s)
log "start on $iid ($host) for $P; the block ends $(date -u -d "@$end" +%FT%TZ)"
cron="@reboot cd $PWD && setsid nohup bash scripts/aws/box_boot.sh $P >>$HOME/box_boot.log 2>&1"
{ crontab -l 2>/dev/null | grep -vF scripts/aws/box_boot.sh; echo "$cron"; } | crontab - || log "no crontab: a reboot will not restart this"

live_workers() {  # ids of the workers that are alive, or whose trainer still is
  cat /proc/[0-9]*/environ 2>/dev/null | tr '\0' '\n' | sed -n 's/^SLURM_JOB_ID=\(box-.*\)$/\1/p' | sort -u
}
status() {
  echo "time $(date -u +%FT%TZ)  instance $iid  block end $(date -u -d "@$end" +%FT%TZ)  box $name"
  echo "markers: $(ls "$mark" | tr '\n' ' ')"
  [[ -s $mark/fetch_failed ]] && echo "NOT FETCHED: $(tr '\n' ' ' <"$mark/fetch_failed")"
  echo "gpu memory: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  echo "disk: $(df -h --output=used,avail,pcent / | tail -1)"
  echo "trainers: $(pgrep -u "$(id -un)" -fc 'python -m rlvr_speedrun')"
  echo "live workers: $(live_workers | tr '\n' ' ')"
  local f
  for f in queue/*.txt; do [[ -e $f ]] && echo "$f: $(wc -l <"$f") lines"; done
  echo "ok $(grep -c ' ok$' queue/done.txt 2>/dev/null)  FAILED $(grep -c ' FAILED$' queue/done.txt 2>/dev/null)"
  echo "running:"
  grep -Hs . queue/running/*
  echo "worker log:"
  tail -6 queue/worker.log 2>/dev/null
  echo "keeper log:"
  tail -6 "$mark/keeper.log" 2>/dev/null
}
push_logs() {
  status >"$mark/status.txt" 2>&1
  aws s3 cp --only-show-errors "$mark/status.txt" "$P/out/status.txt"
  aws s3 cp --only-show-errors "$HOME/box_boot.log" "$P/out/log/$iid/box_boot.log"
  [[ -r /var/log/rlvr-box.log ]] && aws s3 cp --only-show-errors /var/log/rlvr-box.log "$P/out/log/$iid/rlvr-box.log"
}

# --- fetch: the plan, the queues and the files the runs read -----------------------------------------------------
if [[ ! -e $mark/fetched ]]; then
  aws configure set default.s3.max_concurrent_requests 20
  aws configure set default.s3.multipart_chunksize 64MB
  until try aws s3 cp --only-show-errors "$P/in/plan" "$mark/plan"; do  # nothing can run without a plan
    push_logs
    sleep 60
  done
  try aws s3 sync --only-show-errors --delete "$P/in/queue/" queue/staged/ || echo "the queues" >>"$mark/fetch_failed"
  try aws s3 cp --only-show-errors "$P/in/home.txt" "$mark/home.txt" || echo "home.txt" >>"$mark/fetch_failed"
  paths=()
  [[ -e $mark/home.txt ]] && mapfile -t paths <"$mark/home.txt"
  for path in ${paths[@]+"${paths[@]}"}; do
    [[ -n $path ]] || continue
    if [[ $path == *.tar ]]; then  # a directory staged as one tar file (the Hugging Face cache: it holds symlinks)
      path=${path%.tar}
      mkdir -p "$HOME/${path%/*}"
      try aws s3 cp --only-show-errors "$H/$path.tar" "$HOME/$path.tar" </dev/null \
        && tar xf "$HOME/$path.tar" -C "$HOME/${path%/*}" && rm -f "$HOME/$path.tar"
    else
      try aws s3 sync --only-show-errors "$H/$path/" "$HOME/$path/" </dev/null
    fi
    [[ -e $HOME/$path ]] || { echo "$path" >>"$mark/fetch_failed"; log "NOT FETCHED: $path"; }
  done
  touch "$mark/fetched"
  log "fetched: the plan, $(ls queue/staged | wc -l) queues, ${#paths[@]} paths under ~"
fi

PUSH_S=300 STATE_S=3600 KEEPER_S=600 NEED_VENV=1 SHUTDOWN_WHEN_IDLE=0
plan=()
mapfile -t lines <"$mark/plan"
for line in "${lines[@]}"; do
  case $line in
    boot:*) eval "${line#boot:}" ;;
    '' | '#'*) ;;
    *) plan+=("$line") ;;
  esac
done

slots() {  # "<worker id> <number of its plan line> <gpu> <k>" for every worker the plan asks for
  local i=0 line
  for line in "${plan[@]}"; do
    i=$(( i + 1 ))
    (
      eval "$line"
      for k in $(seq "${FIRST:-1}" $(( ${FIRST:-1} + ${PER_GPU:-1} - 1 ))); do
        for g in ${GPUS:?every plan line needs GPUS}; do echo "box-$host-g$g-$k $i $g $k"; done
      done
    )
  done
}
start_line() {  # start the workers of one plan line; further arguments (GPUS=3 PER_GPU=1 FIRST=2) override it
  (
    export SCREEN_RESUME=1 RUNNER=scripts/aws/box_runner.sh
    eval "export $1"
    shift
    (( $# )) && export "$@"
    exec bash scripts/aws/box_workers.sh 9>&- </dev/null
  )
}

# --- restore: carry on from what an earlier instance of this box sent to S3 ---------------------------------------
if [[ ! -e $mark/restored ]]; then
  if list_keys "$key/out/rlvr-screens/queue/snap.tar" | grep -q .; then
    log "an earlier instance left a snapshot: bringing its results back"
    until aws s3 sync --only-show-errors "$P/out/rlvr-screens/" ./; do
      log "could not bring the results back; trying again"
      sleep 20
    done
  fi
  # shellcheck disable=SC2046
  if python3 scripts/aws/box_queue.py restore $(slots | cut -d' ' -f1) >"$mark/restore.out" 2>&1; then
    sed "s/^/$(date -u +%FT%TZ) box_boot: /" "$mark/restore.out"
  else  # should not happen; rather than sit idle, run the staged queues less what the snapshot lists as done
    log "RESTORE FAILED: $(tail -3 "$mark/restore.out" | tr '\n' ' '); falling back to the staged queues"
    cp queue/snap/done.txt queue/done.txt 2>/dev/null
    for f in queue/staged/*.txt; do
      grep -vxFf <(sed -n 's/ ok$//p' queue/done.txt 2>/dev/null) "$f" >"queue/$(basename "$f")"
    done
    rm -f queue/running/*
  fi
  while read -r k; do
    rel=${k#"box-state/$name/"}
    rel=${rel%/state.pt}
    if grep -qxF "${rel%/resume/*} ${rel##*/} ok" queue/done.txt 2>/dev/null; then
      aws s3 rm --recursive --only-show-errors "$S/$rel/" </dev/null
    elif try aws s3 sync --only-show-errors "$S/$rel/" "$CKPT_DIR/$rel/" </dev/null; then
      printf '%s\n%s\n' "$rel" "$(stat -c %Y "$CKPT_DIR/$rel/state.pt")" >"$mark/states/${rel//\//%}"
      log "saved state brought back: $rel"
    fi
  done < <(list_keys "box-state/$name/" | grep '/state\.pt$')
  touch "$mark/restored"
fi

# --- the parts of the main loop -------------------------------------------------------------------------------------
build_venv() {  # cloud-init ended without ~/venv.ok: the same build as the user data, again
  local uv=$HOME/.local/bin/uv py=$HOME/orcd/pool/vllm-venv/bin/python
  log "cloud-init ended without ~/venv.ok: building the venv again"
  [[ -x $uv ]] || curl -LsSf https://astral.sh/uv/install.sh | sh
  [[ -x $py ]] || "$uv" venv -q -p 3.11 "$HOME/orcd/pool/vllm-venv"
  { "$uv" pip install -q --python "$py" -r "$HOME/vllm-venv-freeze.txt" \
    || "$uv" pip install -q --python "$py" "vllm==0.30.0" "transformers==5.17.0" pyarrow numpy accelerate huggingface_hub; } \
    && "$py" -c 'import torch, vllm, transformers; print("torch", torch.__version__, torch.cuda.get_device_name(0))' \
    && touch "$HOME/venv.ok"
}
push() {  # results, then the queue snapshot that describes them, then the logs
  local gone=() f rel
  for f in "$mark"/states/*; do  # states in S3 whose seed has finished: safe to drop once its results are up
    [[ -e $f ]] || continue
    rel=$(head -1 "$f")
    [[ ! -e $CKPT_DIR/$rel/state.pt && -e ${rel%/resume/*}/seeds/${rel##*/}/train_log.jsonl ]] && gone+=("$f")
  done
  flock queue/lock python3 scripts/aws/box_queue.py snap 9>&-
  if aws s3 sync --only-show-errors ./ "$P/out/rlvr-screens/" --exclude '*' \
    --include 'calibration/*/seeds/*' --include 'calibration/*/extra/*' --include 'records/*/seeds/*' \
    --include 'records/*/extra/*' --include 'slurm_logs/*' --include 'queue/worker.log' \
    --exclude '*.pt' --exclude '*.safetensors' --exclude '*.bin'; then
    aws s3 cp --only-show-errors queue/snap.tar "$P/out/rlvr-screens/queue/snap.tar"
    for f in ${gone[@]+"${gone[@]}"}; do
      aws s3 rm --recursive --only-show-errors "$S/$(head -1 "$f")/" && rm -f "$f"
    done
  else
    log "the results push failed; the snapshot in S3 stays as it was"
  fi
  aws s3 sync --only-show-errors "$CKPT_DIR/" "$P/out/rlvr-ckpts/" --exclude '*' --include '*.json' --include '*.jsonl' \
    --exclude '*/pretrained/*' --exclude '*/final/*' --exclude '*/resume/*' --exclude '.box_push/*'
  push_logs
}
push_states() {  # each saved state that changed since it was last sent
  local f rel m rec link=$CKPT_DIR/.box_push/state.pt
  mkdir -p "$CKPT_DIR/.box_push"
  while read -r f; do
    rel=${f#"$CKPT_DIR/"}
    rel=${rel%/state.pt}
    rec=$mark/states/${rel//\//%}
    m=$(stat -c %Y "$f" 2>/dev/null) || continue
    [[ $(sed -n 2p "$rec" 2>/dev/null) == "$m" ]] && continue
    # a hard link keeps this version whole while the trainer replaces state.pt; the logs are read after it, so the
    # logs in S3 are never behind the state (the trainer drops log lines past the state's step when it resumes)
    ln -f "$f" "$link" 2>/dev/null || continue
    m=$(stat -c %Y "$link")
    aws s3 sync --only-show-errors "$CKPT_DIR/$rel/" "$S/$rel/" --exclude '*' --include '*.jsonl' </dev/null \
      && aws s3 cp --only-show-errors "$link" "$S/$rel/state.pt" </dev/null \
      && printf '%s\n%s\n' "$rel" "$m" >"$rec.tmp" && mv "$rec.tmp" "$rec"
    rm -f "$link"
  done < <(find "$CKPT_DIR" -path '*/resume/*' -name state.pt 2>/dev/null)
}
keeper() {  # put each FAILED item back once; restart dead workers that have work
  [[ -e queue/keeper.off ]] && return
  local live id i g k q n spare
  local -A taken=()
  flock queue/lock python3 scripts/aws/box_queue.py retry 9>&- 2>&1 | sed "s/^/$(date -u +%FT%TZ) /" >>"$mark/keeper.log"
  live=$(live_workers)
  while read -r id i g k; do
    grep -qxF "$id" <<<"$live" && continue
    if [[ ! -s queue/running/$id ]]; then  # no item in hand: restart only for a queued item no other restart takes
      q=$(
        eval "${plan[i - 1]}"
        (( (${DEADLINE:-$(( end - 35 * 60 ))} - $(date +%s)) / 60 >= ${RESERVE_MIN:-60} )) || exit 0
        for q in ${QUEUE//:/ }; do [[ -s queue/$q.txt ]] && { echo "$q"; break; }; done
      )
      [[ -n $q ]] || continue
      spare=$(( $(wc -l <"queue/$q.txt") - ${taken[$q]:-0} ))
      (( spare > 0 )) || continue
      taken[$q]=$(( ${taken[$q]:-0} + 1 ))
    fi
    n=$(cat "$mark/restarts/$id" 2>/dev/null || echo 0)
    (( n >= 6 )) && continue
    echo $(( n + 1 )) >"$mark/restarts/$id"
    echo "$(date -u +%FT%TZ) restarting $id (restart $(( n + 1 )) of 6): $(cat "queue/running/$id" 2>/dev/null)" >>"$mark/keeper.log"
    start_line "${plan[i - 1]}" "GPUS=$g" PER_GPU=1 "FIRST=$k" >/dev/null
  done < <(slots)
}
idle() {  # no worker, no item in hand, nothing queued
  [[ -z $(live_workers) ]] || return 1
  grep -qs . queue/running/* && return 1
  local f
  for f in queue/staged/*.txt; do [[ -s queue/$(basename "$f") ]] && return 1; done
  return 0
}

# --- main loop: never exits on an error -----------------------------------------------------------------------------
last_push=0 last_state=0 last_keeper=0 last_venv=0 state_pid=
while true; do
  now=$(date +%s)
  if [[ ! -e $mark/workers_started ]]; then
    if [[ $NEED_VENV == 0 || -e $HOME/venv.ok ]]; then
      log "starting the workers of the plan (${#plan[@]} lines)"
      for line in "${plan[@]}"; do start_line "$line"; done
      touch "$mark/workers_started"
      last_keeper=$(date +%s)
    elif [[ -e /var/lib/cloud/instance/boot-finished ]] && (( now - last_venv >= 600 )); then
      build_venv
      last_venv=$(date +%s)
    fi
  fi
  if (( now - last_push >= PUSH_S )); then
    push
    last_push=$(date +%s)
  fi
  state_s=$STATE_S
  (( end - now < 75 * 60 && state_s > 600 )) && state_s=600  # near the end of the block every saved state counts
  if (( now - last_state >= state_s )) && ! { [[ -n $state_pid ]] && kill -0 "$state_pid" 2>/dev/null; }; then
    push_states 9>&- </dev/null &
    state_pid=$!
    last_state=$now
  fi
  if [[ -e $mark/workers_started ]] && (( now - last_keeper >= KEEPER_S )); then
    keeper
    last_keeper=$(date +%s)
  fi
  if [[ $SHUTDOWN_WHEN_IDLE == 1 && -e $mark/workers_started ]] && idle; then
    keeper  # a failed item may still be owed its one retry
    sleep 5
    if idle; then
      log "idle: last push, then shutdown"
      [[ -n $state_pid ]] && wait "$state_pid"
      push_states
      push
      touch "$mark/finished"
      push_logs
      sudo shutdown -h now
      exit 0
    fi
  fi
  sleep 20
done
