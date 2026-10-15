#!/usr/bin/env bash
# Runs ON the GPU instance (started by scripts/aws/calibrate.sh). Installs the reference
# stack, runs a pipeline smoke test, then the plateau calibration seeds one after another.
# Progress markers go to ~/status.log; ~/DONE is written when everything has finished.
set -uo pipefail
cd ~/rlvr
exec >>~/remote.log 2>&1
status() { echo "$(date -u +%FT%TZ) $*" | tee -a ~/status.log; }

status "setup start on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv -q -p 3.11 ~/venv
source ~/venv/bin/activate
uv pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install -q triton==3.1.0  # torch 2.5.1's compiler backend; not pulled from the cu124 index
uv pip install -q "transformers==4.57.6" numpy accelerate pytest setuptools  # triton 3.1 imports setuptools at runtime; uv venvs lack it
uv pip install -q --no-deps -e .
export PYTHON=python DEVICE=cuda
# fail fast if --compile cannot work. The reference venv (uv's standalone Python) failed
# "no working triton" on the DL AMI twice, so fall back to the AMI's own PyTorch env, which
# AWS ships with a tested torch.compile; result.json records whichever versions ran.
compile_ok() { python -c "import torch; f = torch.compile(lambda x: x * 2); assert f(torch.ones(4, device='cuda')).sum().item() == 8" >>~/compile_check.log 2>&1; }
if ! compile_ok; then
  status "reference venv: torch.compile FAILED ($(python -c 'import triton; print("triton", triton.__version__)' 2>&1 | tail -1); cc: $(command -v cc || echo none)); falling back to /opt/pytorch"
  python - >>~/compile_check.log 2>&1 <<'PY' || true
import traceback
try:
    from triton.runtime.driver import driver
    driver.active.get_current_device()
except Exception:
    traceback.print_exc()
PY
  deactivate
  source /opt/pytorch/bin/activate
  pip install -q "transformers==4.57.6" numpy accelerate pytest
  pip install -q --no-deps -e .
  compile_ok || { status "torch.compile FAILED in /opt/pytorch too; aborting (see compile_check.log)"; touch ~/DONE; exit 1; }
fi
status "torch.compile works: $(python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)')"
python -m pytest -q tests >>~/status.log 2>&1 && status "unit tests pass" || status "UNIT TESTS FAILED (continuing)"

# a seed that runs out of memory on a smaller GPU is retried with smaller batches; both only
# change how the same computation is chunked (micro_batch_size) or padded (eval_batch_size)
run() {
  local record=$1 seed=$2; shift 2
  if bash scripts/run_seed.sh "$record" "$seed" "$@"; then status "ok $record seed $seed"; return 0; fi
  if grep -q "OutOfMemoryError\|CUDA out of memory" "$record/seeds/$seed/stdout.log"; then
    status "OOM $record seed $seed -> retry with --micro-batch-size 32 --eval-batch-size 250"
    mkdir -p "$record/extra/oom_$seed" && mv "$record/seeds/$seed"/* "$record/extra/oom_$seed/"
    bash scripts/run_seed.sh "$record" "$seed" "$@" --micro-batch-size 32 --eval-batch-size 250 && { status "ok $record seed $seed (small batches)"; return 0; }
  fi
  status "FAILED $record seed $seed"; return 1
}

# 1. pipeline smoke test: 3 training steps, then the full probe and full confirmation eval
smoke=~/smoke/track_a/006_smoke
mkdir -p "$smoke"
python - "$smoke/config.json" <<'EOF'
import json, sys
c = json.load(open("calibration/plateau_005/config.json"))
c.update(max_steps=3, eval_start_step=1, eval_every=3, eval_limit=200)
json.dump(c, open(sys.argv[1], "w"), indent=2)
EOF
if ! run "$smoke" 0; then status "smoke test failed; aborting before the plateau runs"; touch ~/DONE; exit 1; fi
python -c "import json;print(json.load(open('$smoke/seeds/0/confirm.json'))['confirm'])" >>~/status.log

# 2. plateau calibration seeds
for seed in ${SEEDS:-0 1 2}; do
  run calibration/plateau_005 "$seed"
done
status "all done"
touch ~/DONE
