#!/usr/bin/env bash
# Runner for research queue workers (queue_worker.sbatch with RUNNER=scripts/research/run_item.sh).
# An item "<dir> <seed>" is
#   - a record seed if <dir> is under records/v3/: STAGE=all scripts/run_seed_v3.sh, as a record
#     worker runs it (timed training, then the test ladder), with the token file copied to node-local
#     disk for track D;
#   - a dev-ladder sampler eval if <dir>/eval.json exists (below);
#   - a track D screen with its own pretraining if <dir>/config.json has a v3 "pretrain" block and no
#     "screen_pretrained": STAGE=train scripts/run_seed_v3.sh (pretraining + GRPO with in-loop dev
#     evals; the test-ladder stage never runs), with the token file copied to node-local disk;
#   - else a GRPO screen (scripts/slurm/screen_seed.sh, unchanged).
#
# eval.json: {"model": "<checkpoint dir; {seed} is replaced by the item's seed>", "temperature": 1.0,
#             "constrain": true, "think_budget": 0, "limit_per_level": 0}
# The eval writes <dir>/seeds/<seed>/dev_confirm.json with scripts/research/dev_ladder_eval.py (the
# reference scorer on the dev ladder; the test ladder is never read).
set -euo pipefail
dir=${1:?usage: run_item.sh <dir> <seed>}
seed=${2:?usage: run_item.sh <dir> <seed>}
PYTHON=${PYTHON:-python}
# Untimed warm start first (scripts/warm_start.sh; nothing touches the GPU): it reads the environment and
# the base model into the node's page cache, which the CUDA check and the trainer otherwise read cold over
# the network filesystems. On for screens and evals; for record seeds only with WARM_START=1 (QUESTIONS
# 2026-10-26 07:05). Its time goes to worker.log. On a broken GPU, a cold warm start can make the first
# failure slow (over 5 min), but the next ones are quick, so the worker still stops.
warm=${WARM_START:-1}
[[ $dir == records/v3/* ]] && warm=${WARM_START:-0}
if [[ -n ${SLURM_JOB_ID:-} && $warm == 1 ]]; then
  msg=$(timeout 900 bash scripts/warm_start.sh "$dir/config.json" 2>&1 | tail -n 1) || true
  echo "$(date -u +%FT%TZ) worker $SLURM_JOB_ID: ${msg:-warm start timed out} on $(hostname) before $dir $seed" | tee -a queue/worker.log >&2
fi
# A GPU that nvidia-smi lists but CUDA can't open ("busy or unavailable": one of node2801's on
# 2026-10-24) fails every item at its first CUDA call, and Slurm hands the freed GPU to the next job.
# Try CUDA first. If that fails with a CUDA error, put the item back at the front of the worker's
# first queue and empty the worker's in-hand file, so the worker's exit after two quick failures
# consumes nothing (its FAILED line in done.txt is then only a note). The timeout keeps both tries
# under its 5-min mark. Any other failure only gets a note and the item runs: on node4109 on
# 2026-10-24 the check timed out with no output, and putting the item back would just stop a worker.
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  rc=0; err=$(timeout 150 $PYTHON -c 'import torch; torch.zeros(1, device="cuda"); torch.cuda.synchronize()' 2>&1) || rc=$?
  if (( rc != 0 )) && grep -q 'CUDA' <<<"$err"; then
    q=${QUEUE:-pending}; q=queue/${q%%:*}.txt
    flock queue/lock bash -c 'touch "$0"; { echo "$1"; cat "$0"; } >"$0.new" && mv "$0.new" "$0"' "$q" "$dir $seed"
    : >"queue/running/$SLURM_JOB_ID"
    msg=$(grep -m1 'Error' <<<"$err" || tail -n 1 <<<"$err")
    echo "$(date -u +%FT%TZ) worker $SLURM_JOB_ID: CUDA can't use the GPU on $(hostname) (${msg:0:160}); put $dir $seed back at the front of $q" | tee -a queue/worker.log >&2
    exit 1
  fi
  if (( rc != 0 )); then
    echo "$(date -u +%FT%TZ) worker $SLURM_JOB_ID: CUDA check on $(hostname) exited $rc without a CUDA error (124 = timed out); running $dir $seed anyway" | tee -a queue/worker.log >&2
  fi
fi
if [[ $dir == records/v3/* ]]; then
  # Slurm may start a preemptable worker in a backfill gap as short as its --time-min, and the
  # worker redoes the item in hand after a preemption without checking the time left. A record seed
  # that can't finish (its budgets + 30 min for loading, the data copy and the test ladder) isn't
  # started: the item fails with a note in worker.log, to be requeued by hand.
  need=$($PYTHON -c "import json,sys; c=json.load(open(sys.argv[1])); print(int(c['v3'].get('pretrain', {}).get('minutes', 0) + c.get('time_budget_s', 0) / 60 + 30))" "$dir/config.json")
  l=$([[ -n ${SLURM_JOB_ID:-} ]] && squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null || true)
  if [[ -n $l && $l != UNLIMITED ]]; then  # %L is [D-]HH:MM:SS or MM:SS
    d=0 h=0 m=0
    [[ $l == *-* ]] && { d=${l%%-*}; l=${l#*-}; }
    IFS=: read -ra p <<<"$l"
    case ${#p[@]} in 3) h=${p[0]} m=${p[1]} ;; 2) m=${p[0]} ;; esac
    left=$(( 10#$d * 1440 + 10#$h * 60 + 10#$m ))
    if (( left < need )); then
      echo "$(date -u +%FT%TZ) worker $SLURM_JOB_ID: $left min left but $dir $seed needs $need; not started, requeue it" | tee -a queue/worker.log >&2
      exit 1
    fi
  fi
  # A record seed stopped during the data copy or pretraining leaves only stdout.log in its seed dir,
  # which the worker's restart check doesn't look for: keep that run under extra/ as the worker does
  # for later stops, instead of letting the rerun overwrite its log.
  if [[ -e "$dir/seeds/$seed/stdout.log" ]]; then
    n=1; while [[ -e "$dir/extra/interrupted_${seed}_$n" ]]; do n=$((n + 1)); done
    mkdir -p "$dir/extra" && mv "$dir/seeds/$seed" "$dir/extra/interrupted_${seed}_$n"
    echo "run_item.sh: $dir seed $seed stopped before its pretraining log; kept under extra/interrupted_${seed}_$n" >&2
  fi
  STAGE=all LOCAL_DATA_DIR=${LOCAL_DATA_DIR:-/tmp} exec bash scripts/run_seed_v3.sh "$dir" "$seed"
fi
if [[ ! -f "$dir/eval.json" ]]; then
  pretrain=$($PYTHON -c "import json,sys; c=json.load(open(sys.argv[1])); print(int('pretrain' in c.get('v3', {}) and not c.get('screen_pretrained')))" "$dir/config.json")
  [[ $pretrain == 1 ]] && STAGE=train LOCAL_DATA_DIR=${LOCAL_DATA_DIR:-/tmp} exec bash scripts/run_seed_v3.sh "$dir" "$seed"
  exec bash scripts/slurm/screen_seed.sh "$dir" "$seed"
fi
out="$dir/seeds/$seed"
[[ -e "$out/dev_confirm.json" ]] && { echo "$out already holds an eval" >&2; exit 1; }
mkdir -p "$out"
mapfile -t args < <($PYTHON - "$dir/eval.json" "$seed" <<'EOF'
import json, os, sys
c = json.load(open(sys.argv[1]))
args = ["--model", os.path.expandvars(os.path.expanduser(c["model"].format(seed=sys.argv[2]))),
        "--temperature", str(c.get("temperature", 1.0))]
if c.get("constrain"):
    args += ["--constrain", "--think-budget", str(c.get("think_budget", 0))]
if c.get("limit_per_level"):
    args += ["--limit-per-level", str(c["limit_per_level"])]
print("\n".join(args))
EOF
)
{
  echo "=== dev-ladder eval $dir seed $seed on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) $(date -u +%FT%TZ)"
  $PYTHON scripts/research/dev_ladder_eval.py "${args[@]}" --out "$out/dev_confirm.json"
  echo "=== done $(date -u +%FT%TZ)"
} 2>&1 | tee "$out/stdout.log"
