#!/usr/bin/env bash
# Packing probe for one GPU: for each K in KS, start K GRPO runs of CONFIG at once (STAGGER_S apart, so each
# vLLM engine sizes its cache on its own), each for STEPS steps from MODEL at constant lr, no weights saved,
# vllm_gpu_memory MEM_<K>, in the queue workers' environment. Then print each run's mean s/step over steps
# 6..STEPS (train_s, so evals are excluded) and the summed steps per hour, to pick PER_GPU for a box
# (scripts/aws/box_workers.sh). Track A runs start from r7's searching-regime weights (long outputs), track D
# runs from D 002's pretrained seed 0. Dev ladder only (a short final eval). Logs: $OUT/<tag>_k<K>_<i>/.
#
#   sbatch -p mit_normal_gpu --gres=gpu:b200:1 --time=02:30:00 -c 16 --mem=200G -o slurm_logs/probe/%j.out \
#     --export=ALL,RLVR_DIR=$HOME/rlvr-screens scripts/research/pack_probe.sh
set -uo pipefail
cd "${RLVR_DIR:-$HOME/rlvr-screens}"
source "${VENV:-$HOME/orcd/pool/vllm-venv}/bin/activate"
command -v module >/dev/null && module load cuda/13.0.1  # vLLM JIT-compiles kernels with nvcc (on Engaging; a box has none)
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0  # as queue_worker.sbatch
OUT=${OUT:-$HOME/orcd/pool/rlvr-ckpts/probe/pack_$(date +%m%d_%H%M)_${SLURM_JOB_ID:-local}}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
gcc_ok() { printf '#include <stdlib.h>\nint main(void){return 0;}\n' | gcc -x c -o /dev/null - 2>/dev/null; }
gcc_ok || export C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu CPLUS_INCLUDE_PATH=/usr/include/x86_64-linux-gnu LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

phase() {  # tag config model steps K mem [extra grpo args...]
  local tag=$1 config=$2 model=$3 steps=$4 k=$5 mem=$6
  shift 6
  local pids=()
  for i in $(seq 1 "$k"); do
    local dir=$OUT/${tag}_k${k}_$i
    mkdir -p "$dir"
    python -m rlvr_speedrun.grpo --config "$config" --device cuda --model "$model" --max-steps "$steps" \
      --time-budget-s 0 --lr-schedule constant --eval-start-step 100000000 --eval-limit 60 --no-save \
      --vllm-gpu-memory "$mem" --seed "$i" --out-dir "$dir" "$@" >"$dir/stdout.log" 2>&1 &
    pids+=($!)
    (( i < k )) && sleep "${STAGGER_S:-90}"
  done
  for p in "${pids[@]}"; do wait "$p"; done
  python - "$OUT" "${tag}_k${k}" "$k" <<'EOF'
import json, sys
from pathlib import Path
out, prefix, k = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
total = 0.0
for i in range(1, k + 1):
    path = out / f"{prefix}_{i}" / "train_log.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    rows = [r for r in rows if r["step"] >= 6]
    if len(rows) < 2:
        print(f"{prefix}_{i}: FAILED (see stdout.log)")
        continue
    sps = (rows[-1]["train_s"] - rows[0]["train_s"]) / (rows[-1]["step"] - rows[0]["step"])
    mem = max(r.get("gpu_mem_gb", 0) for r in rows)
    samp = sum(r["t_sample"] for r in rows) / len(rows)
    upd = sum(r["t_update"] for r in rows) / len(rows)
    tok = sum(r["tokens/s"] * (r["t_sample"] + r["t_update"]) for r in rows) / len(rows) / 512
    total += 3600 / sps
    print(f"{prefix}_{i}: {sps:6.2f} s/step (sample {samp:5.2f}, update {upd:5.2f}), ~{tok:4.0f} tokens/sample, gpu mem seen {mem:5.1f} GB")
print(f"{prefix}: {k} runs, {total:7.1f} steps/h summed")
EOF
}

A_CFG=calibration/research_r7/a01_fp32_lr1e-6_skip_cos_mal0_s300/config.json
A_MODEL=${A_MODEL:-$HOME/orcd/pool/rlvr-ckpts/calibration/research_r7/a01_fp32_lr1e-6_skip_cos_mal0_s300/seeds/1/final}
D_CFG=records/v3/track_d/002_dm_math_fp32/config.json
D_MODEL=${D_MODEL:-$HOME/orcd/pool/rlvr-ckpts/records/v3/track_d/002_dm_math_fp32/seeds/0/pretrained}
for k in ${KS:-1 2}; do
  mem_var=MEM_$k
  phase a "$A_CFG" "$A_MODEL" "${A_STEPS:-30}" "$k" "${!mem_var:-0.2}" --micro-batch-tokens "${A_MBT:-4096}"
done
for k in ${D_KS:-1 4}; do
  phase d "$D_CFG" "$D_MODEL" "${D_STEPS:-300}" "$k" "${D_MEM:-0.08}"
done
echo "probe done $(date -u +%FT%TZ)"
