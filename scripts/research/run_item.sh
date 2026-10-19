#!/usr/bin/env bash
# Runner for research queue workers (queue_worker.sbatch with RUNNER=scripts/research/run_item.sh):
# an item "<dir> <seed>" is a dev-ladder sampler eval if <dir>/eval.json exists, else a GRPO
# screen (scripts/slurm/screen_seed.sh, unchanged).
#
# eval.json: {"model": "<checkpoint dir; {seed} is replaced by the item's seed>", "temperature": 1.0,
#             "constrain": true, "think_budget": 0, "limit_per_level": 0}
# The eval writes <dir>/seeds/<seed>/dev_confirm.json with scripts/research/dev_ladder_eval.py (the
# reference scorer on the dev ladder; the test ladder is never read).
set -euo pipefail
dir=${1:?usage: run_item.sh <dir> <seed>}
seed=${2:?usage: run_item.sh <dir> <seed>}
[[ -f "$dir/eval.json" ]] || exec bash scripts/slurm/screen_seed.sh "$dir" "$seed"
PYTHON=${PYTHON:-python}
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
