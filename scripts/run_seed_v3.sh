#!/usr/bin/env bash
# Run one seed of a benchmark v3 record end to end (docs/RULES_v3.md):
#
#   bash scripts/run_seed_v3.sh records/v3/track_{a,d}/<NNN>_<slug> <SEED>
#
# The record's config.json holds the trainer config plus a "v3" block:
#   {"track": "A" | "D",
#    "pretrain": {"minutes": 112, "tokens": <.bin>, "tokenizer": <dir>, "args": "<extra pretrain_scratch.py flags>"},  # track D only
#    "sampler": {"temperature": 1.0, "constrain": true, "think_budget": 0}}
# Track D pretrains with --seed SEED (timed), then trains with GRPO from that model (timed), then
# scores the final checkpoint on the test ladder with the record's sampler (untimed). Writes
# <record>/seeds/<SEED>/{train_log,eval_log,result,confirm_v3,clock}.json(l) and stdout.log;
# weights go to $CKPT_DIR/<record>/seeds/<SEED>/{pretrained,final} (default ~/orcd/pool/rlvr-ckpts).
#
# clock.json: GPU name, pretraining wall time, GRPO process wall time minus in-loop eval time, and
# their sum versus the track budget (benchmark.V3_TIME_BUDGET_S). Process start-up, model loading,
# vLLM initialisation and the final save all count.
#
# STAGE=train runs only the timed part, STAGE=confirm only the (untimed) test-ladder scoring, so the
# two can be separate Slurm jobs; the default runs both. A seed directory that already holds a run
# is never overwritten: move it to <record>/extra/ first (RULES_v3: every run is reported).
# LOCAL_DATA_DIR=<node-local dir> copies the track D token file there first (untimed data
# preparation): random reads from a network filesystem made pretraining ~9x slower on Engaging.
set -euo pipefail

record=${1:?usage: run_seed_v3.sh <record dir> <seed>}
seed=${2:?usage: run_seed_v3.sh <record dir> <seed>}
PYTHON=${PYTHON:-python}
out="$record/seeds/$seed"
weights="${CKPT_DIR:-$HOME/orcd/pool/rlvr-ckpts}/$record/seeds/$seed"
mkdir -p "$out" "$weights"
cfg() { $PYTHON -c "import json,sys; c=json.load(open(sys.argv[1]))['v3']; v=eval(sys.argv[2], {}, c); print(json.dumps(v) if isinstance(v, (bool, dict, list)) else v)" "$record/config.json" "$1"; }
track=$(cfg track)
now() { date +%s.%N; }
STAGE=${STAGE:-all}
if [[ $STAGE != confirm ]] && [[ -e "$out/train_log.jsonl" || -e "$out/pretrain_log.jsonl" ]]; then
  echo "$out already holds a run; move it to $record/extra/ first" >&2; exit 1
fi

if [[ $STAGE != confirm ]]; then {
  echo "=== v3 track $track seed $seed on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) $(date -u +%FT%TZ)"
  pre_s=0
  model_flag=()
  if [[ "$track" == D ]]; then
    tokens=$(cfg "pretrain['tokens']")
    if [[ -n "${LOCAL_DATA_DIR:-}" ]]; then
      echo "=== copy $tokens to $LOCAL_DATA_DIR (untimed)"
      local_dir="$LOCAL_DATA_DIR/rlvr-${SLURM_JOB_ID:-$$}"  # per job: two seeds may share a node
      mkdir -p "$local_dir" && cp "$tokens" "$local_dir/" && tokens="$local_dir/$(basename "$tokens")"
    fi
    echo "=== pretrain (timed)"
    t0=$(now)
    # shellcheck disable=SC2046
    $PYTHON scripts/pretrain_scratch.py train --tokens "$tokens" --tokenizer "$(cfg "pretrain['tokenizer']")" \
      --out "$weights/pretrained" --minutes "$(cfg "pretrain['minutes']")" --seed "$seed" $(cfg "pretrain.get('args', '')")
    pre_s=$($PYTHON -c "print($(now) - $t0)")
    cp "$weights/pretrained/pretrain_log.jsonl" "$out/pretrain_log.jsonl"
    [[ -n "${local_dir:-}" ]] && rm -rf "$local_dir"
    model_flag=(--model "$weights/pretrained")
  fi
  echo "=== GRPO (timed)"
  t0=$(now)
  $PYTHON -m rlvr_speedrun.grpo --config "$record/config.json" --device cuda --seed "$seed" --out-dir "$out" ${model_flag[@]+"${model_flag[@]}"}
  rl_wall=$($PYTHON -c "print($(now) - $t0)")
  $PYTHON - "$out" "$track" "$pre_s" "$rl_wall" <<'EOF'
import json, subprocess, sys
from rlvr_speedrun import benchmark
out, track, pre_s, rl_wall = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
last = json.loads(open(f"{out}/eval_log.jsonl").readlines()[-1])
eval_s = last["wall_s"] - last["train_s"]
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.splitlines()[0]
total = pre_s + rl_wall - eval_s
clock = {"gpu": gpu, "pretrain_s": pre_s, "grpo_wall_s": rl_wall, "grpo_eval_s": eval_s, "total_s": total,
         "budget_s": benchmark.V3_TIME_BUDGET_S[track], "within_budget": total <= benchmark.V3_TIME_BUDGET_S[track]}
open(f"{out}/clock.json", "w").write(json.dumps(clock, indent=1) + "\n")
print("clock", clock)
EOF
  rm -rf "${weights:?}/final" && mv "$out/final" "$weights/final"
} 2>&1 | tee "$out/stdout.log"; fi

if [[ $STAGE != train ]]; then {
  echo "=== test ladder (untimed) on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) $(date -u +%FT%TZ)"
  sampler=(--temperature "$(cfg "sampler.get('temperature', 1.0)")")
  [[ "$(cfg "sampler.get('constrain', False)")" == true ]] && sampler+=(--constrain --think-budget "$(cfg "sampler.get('think_budget', 0)")")
  $PYTHON -m rlvr_speedrun.ladder_confirm --model "$weights/final" --out "$out/confirm_v3.json" "${sampler[@]}" ${CONFIRM_LIMIT:+--limit-per-level "$CONFIRM_LIMIT"}
} 2>&1 | tee -a "$out/stdout.log"; fi
