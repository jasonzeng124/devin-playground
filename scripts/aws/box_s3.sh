#!/usr/bin/env bash
# Engaging side of a self-running AWS box (scripts/aws/box_boot.sh runs on the box): puts what the box fetches at
# boot into S3, checks it, and brings back what the box sent. Run from the repo root.
#
#   bash scripts/aws/box_s3.sh stage <name> <commit> <plan file> <queue file>...
#       code.tgz of <commit> (the files scripts/aws/box_sync.sh copies), the plan, the queues (as they are in the
#       working tree) and home.txt: the paths under ~ that the queued configs read. HOME_EXTRA="path ..." adds more.
#   bash scripts/aws/box_s3.sh home <name>     upload the paths of home.txt that S3 lacks. Run it as a CPU job:
#       sbatch -p mit_normal --time=02:00:00 -c 4 --mem=8G --chdir=$PWD --wrap "bash scripts/aws/box_s3.sh home <name>"
#   bash scripts/aws/box_s3.sh check <name>    every queued config is in the code, every file it reads is in S3, the
#       plan gives no worker slot twice; prints the workers per GPU for the memory check. Exit 1 on any problem.
#   bash scripts/aws/box_s3.sh pull <name> <tag> [dir ...]   results into the checkout and $CKPT_DIR, laid out as
#       scripts/aws/box_sync.sh from <tag> does. Default dirs: the parents of the queued arms. STATE=1 also brings
#       the saved state of unfinished seeds (large for track A, and data leaving AWS is billed: only when needed).
#   bash scripts/aws/box_s3.sh status <name>   the box's last status report and its age
#   bash scripts/aws/box_s3.sh clear <name>    remove a TEST box's results and states from S3 (name must contain "test")
#
# <name> is rlvr-box-<reservation id>. In the bucket rlvr-speedrun-<account id> (BUCKET overrides):
#   box/<name>/in/    code.tgz commit plan queue/<q>.txt home.txt        box/<name>/out/   written by the box
#   box/home/         the files under ~ that boxes read, shared by all    box-state/<name>/ saved states, written by the box
set -euo pipefail
cmd=${1:?usage: box_s3.sh stage|home|check|pull|status|clear <name> ...} name=${2:?box name}
shift 2
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
BUCKET=${BUCKET:-rlvr-speedrun-$(aws sts get-caller-identity --query Account --output text)}
P=s3://$BUCKET/box/$name H=s3://$BUCKET/box/home S=s3://$BUCKET/box-state/$name
CKPT=${CKPT_DIR:-$HOME/orcd/pool/rlvr-ckpts}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

in_s3() { aws s3api head-object --bucket "$BUCKET" --key "$1" >/dev/null 2>&1; }
items() { grep -hv -e '^$' -e '^#' "$@" || true; }  # the "<config dir> <seed>" lines of queue files
home_paths() {  # <commit> <queue file>...: the paths under ~ that the queued runs read, and warnings on stderr
  local commit=$1 dir seed cfg pre model tok
  shift
  while read -r dir seed; do
    if ! cfg=$(git show "$commit:$dir/config.json" 2>/dev/null); then
      echo "WARNING: $dir has no config.json in $commit" >&2
      continue
    fi
    pre=$(jq -r '.screen_pretrained // empty' <<<"$cfg")
    model=$(jq -r '.model // empty' <<<"$cfg")
    tok=$(jq -r '.v3.pretrain.tokenizer // empty' <<<"$cfg")
    [[ -n $pre ]] && echo "orcd/pool/rlvr-ckpts/$pre/seeds/$seed/pretrained"
    [[ $model == */* && $model != /* ]] && echo ".cache/huggingface/hub/models--${model//\//--}.tar"
    [[ $tok == "$HOME"/* ]] && echo "${tok#"$HOME"/}"
    if [[ $dir == records/* || -z $pre && $(jq 'has("v3") and (.v3 | has("pretrain"))' <<<"$cfg") == true ]]; then
      echo "WARNING: $dir $seed is not a screen from a saved checkpoint: the box setup does not stage its inputs" >&2
    fi
  done < <(items "$@")
  for dir in ${HOME_EXTRA:-}; do echo "$dir"; done
}

case $cmd in
  stage)
    commit=$(git rev-parse --verify "${1:?commit}^{commit}") plan=${2:?plan file}
    shift 2
    (( $# )) || { echo "no queue files" >&2; exit 1; }
    { git ls-tree -r --name-only "$commit" -- rlvr_speedrun scripts tests data pyproject.toml setup.py uv.lock
      git ls-tree -r --name-only "$commit" -- calibration records \
        | grep -E '(^|/)(config|eval)\.json$' | grep -v -e '/seeds/' -e '/extra/'; } >"$tmp/files"
    # shellcheck disable=SC2046
    git archive --format=tar.gz -o "$tmp/code.tgz" "$commit" -- $(cat "$tmp/files")
    mkdir "$tmp/queue"
    for q in "$@"; do cp "$q" "$tmp/queue/$(basename "$q")"; done
    home_paths "$commit" "$tmp"/queue/*.txt | sort -u >"$tmp/home.txt"
    echo "$commit" >"$tmp/commit"
    if in_s3 "box/$name/out/rlvr-screens/queue/snap.tar"; then
      echo "NOTE: a box has already run as $name; its next instance takes these queues less what its snapshot lists as done" >&2
    fi
    aws s3 sync --only-show-errors --delete "$tmp/queue/" "$P/in/queue/"
    for f in home.txt plan commit code.tgz; do  # code.tgz last: the box starts nothing without it
      src=$tmp/$f
      [[ $f == plan ]] && src=$plan
      aws s3 cp --only-show-errors "$src" "$P/in/$f"
    done
    echo "staged $name: commit ${commit:0:7}, $(wc -l <"$tmp/files") files in code.tgz ($(stat -c %s "$tmp/code.tgz") bytes)," \
      "$(items "$tmp"/queue/*.txt | wc -l) items in $# queues, $(wc -l <"$tmp/home.txt") paths in home.txt"
    ;;
  home)
    aws s3 cp --only-show-errors "$P/in/home.txt" "$tmp/home.txt"
    aws configure set default.s3.max_concurrent_requests 16
    while read -r path; do
      [[ -n $path ]] || continue
      if [[ $path == *.tar ]]; then  # a directory with symlinks (the Hugging Face cache) goes up as one tar file
        in_s3 "box/home/$path" && continue
        dir=$HOME/${path%.tar}
        tar cf - -C "${dir%/*}" "${dir##*/}" | aws s3 cp --only-show-errors - "$H/$path"
      else
        [[ -d $HOME/$path ]] || { echo "MISSING on Engaging: ~/$path" >&2; continue; }
        aws s3 sync --only-show-errors "$HOME/$path/" "$H/$path/"
      fi
      echo "in S3: $path"
    done <"$tmp/home.txt"
    ;;
  check)
    bad=0
    aws s3 sync --only-show-errors "$P/in/" "$tmp/in/" --exclude code.tgz
    commit=$(cat "$tmp/in/commit")
    in_s3 "box/$name/in/code.tgz" || { echo "PROBLEM: no code.tgz"; bad=1; }
    for f in scripts/aws/box_boot.sh scripts/aws/box_queue.py scripts/aws/box_runner.sh scripts/aws/box_workers.sh; do
      git cat-file -e "$commit:$f" 2>/dev/null || { echo "PROBLEM: $f is not in commit ${commit:0:7}"; bad=1; }
    done
    while read -r dir seed; do
      git cat-file -e "$commit:$dir/config.json" 2>/dev/null || { echo "PROBLEM: $dir/config.json is not in commit ${commit:0:7}"; bad=1; }
      [[ -e $dir/seeds/$seed/train_log.jsonl ]] && echo "note: $dir $seed already has results on Engaging"
    done < <(items "$tmp"/in/queue/*.txt)
    dup=$(items "$tmp"/in/queue/*.txt | sort | uniq -d)
    [[ -z $dup ]] || { echo "PROBLEM: queued twice: $dup"; bad=1; }
    home_paths "$commit" "$tmp"/in/queue/*.txt 2>"$tmp/warn" | sort -u >"$tmp/need"
    sed 's/^WARNING/PROBLEM/' "$tmp/warn"
    [[ -s $tmp/warn ]] && bad=1
    cmp -s "$tmp/need" "$tmp/in/home.txt" || { echo "PROBLEM: home.txt differs from what the queued configs read (stage again)"; bad=1; }
    while read -r path; do
      key=box/home/$path
      [[ $path == *.tar ]] || key=$(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "box/home/$path/" --max-items 1 \
        --query 'Contents[0].Key' --output text | head -1)
      [[ $path == */pretrained ]] && key=box/home/$path/model.safetensors
      in_s3 "$key" || { echo "PROBLEM: not in S3: box/home/$path (run: box_s3.sh home $name)"; bad=1; }
    done <"$tmp/in/home.txt"
    # the plan: one worker per (GPU, k); every queue it names is staged
    grep -v -e '^$' -e '^#' -e '^boot:' "$tmp/in/plan" >"$tmp/plan" || true
    while IFS= read -r line; do
      (
        eval "$line"
        : "${GPUS:?a plan line without GPUS}" "${QUEUE:?a plan line without QUEUE}"
        for q in ${QUEUE//:/ }; do [[ -e $tmp/in/queue/$q.txt ]] || echo "PROBLEM: the plan names queue $q, which is not staged" >&2; done
        for k in $(seq "${FIRST:-1}" $(( ${FIRST:-1} + ${PER_GPU:-1} - 1 ))); do
          for g in $GPUS; do echo "$g $k $QUEUE"; done
        done
      ) 2>>"$tmp/planerr" || echo "PROBLEM: bad plan line: $line" >>"$tmp/planerr"
    done <"$tmp/plan" >"$tmp/slots"
    [[ -s $tmp/planerr ]] && { cat "$tmp/planerr"; bad=1; }
    dup=$(cut -d' ' -f1,2 "$tmp/slots" | sort | uniq -d)
    [[ -z $dup ]] || { echo "PROBLEM: worker slot (gpu k) given twice: $dup"; bad=1; }
    [[ -s $tmp/slots ]] || { echo "PROBLEM: the plan starts no workers"; bad=1; }
    echo "commit ${commit:0:7}; $(items "$tmp"/in/queue/*.txt | wc -l) items; $(wc -l <"$tmp/in/home.txt") paths under ~; workers per GPU (check the memory):"
    sort -n "$tmp/slots" | awk '{n[$1]++; q[$1] = q[$1] " " $3} END {for (g in n) print "  GPU " g ": " n[g] " workers:" q[g]}' | sort
    grep '^boot:' "$tmp/in/plan" || true
    (( bad == 0 )) && echo "check ok" || { echo "check FAILED"; exit 1; }
    ;;
  pull)
    tag=${1:?tag}
    shift
    if (( $# == 0 )); then
      aws s3 sync --only-show-errors "$P/in/queue/" "$tmp/queue/"
      # shellcheck disable=SC2046
      set -- $(items "$tmp"/queue/*.txt | cut -d' ' -f1 | xargs -r -n1 dirname | sort -u)
    fi
    for dir in "$@"; do  # always one directory at a time: never a sync at the root of the checkpoints
      mkdir -p "$dir" "$CKPT/$dir"
      aws s3 sync --only-show-errors "$P/out/rlvr-screens/$dir/" "$dir/" --exclude '*' --include '*.json' --include '*.jsonl' --include '*.log'
      if [[ -n ${STATE:-} ]]; then  # the state first and its logs after, so the logs are never behind the state
        aws s3 sync --only-show-errors "$S/$dir/" "$CKPT/$dir/" --exclude '*' --include '*/state.pt'
        aws s3 sync --only-show-errors "$S/$dir/" "$CKPT/$dir/" --exclude '*' --include '*.jsonl'
      fi
      aws s3 sync --only-show-errors "$P/out/rlvr-ckpts/$dir/" "$CKPT/$dir/" --exclude '*' --include '*.json' --include '*.jsonl'
    done
    aws s3 cp --only-show-errors "$P/out/rlvr-screens/queue/worker.log" "queue/${tag}_worker.log" || true
    if aws s3 cp --only-show-errors "$P/out/rlvr-screens/queue/snap.tar" "$tmp/snap.tar"; then
      tar xf "$tmp/snap.tar" -C "$tmp" && cp "$tmp/snap/done.txt" "queue/${tag}_done.txt" 2>/dev/null || true
      [[ -s $tmp/snap/running.txt ]] && sed 's/^/still running at the last snapshot: /' "$tmp/snap/running.txt"
    fi
    mkdir -p "slurm_logs/$tag"
    aws s3 sync --only-show-errors "$P/out/rlvr-screens/slurm_logs/" "slurm_logs/$tag/" || true
    ;;
  status)
    aws s3api head-object --bucket "$BUCKET" --key "box/$name/out/status.txt" --query LastModified --output text \
      | sed 's/^/status written /' || { echo "no status yet"; exit 1; }
    aws s3 cp --only-show-errors "$P/out/status.txt" -
    ;;
  clear)
    [[ $name == *test* ]] || { echo "clear is for test boxes only" >&2; exit 1; }
    aws s3 rm --recursive --only-show-errors "$P/out/"
    aws s3 rm --recursive --only-show-errors "$S/"
    ;;
  *)
    echo "unknown command $cmd" >&2
    exit 1
    ;;
esac
