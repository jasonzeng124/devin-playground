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
if [[ $dir == records/v3/* ]]; then
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
