#!/bin/bash
# r135 (LOG 2026-11-25): compile_update inside update_capture on one B200. Under INBOX c5e94e0 the change leaves
# training the same, so it needs an equivalence check and a benchmark only:
#  1. scripts/research/check_capture_compile.py on D 007 seed 0's final weights (rule in its docstring and the LOG);
#  2. profile_update.py on r134's setup (r106's final_native_ctl, capture with the fine steps, the three draw options
#     on) with compile_update on, 300 timed updates, for the wall step beside r134's and the update's busy time.
# Submit from slurm_logs inside `bash -lc`, e.g.
#   sbatch -p mit_preemptable --time=01:30:00 --gres=gpu:b200:1 -c 16 --mem=96G \
#     --exclude=node5500-c1,node5800-c1,node5702-c1 -J rlvr-prep \
#     -o /home/zjason/rlvr-screens/slurm_logs/r135_%j.out scripts/research/r135_capture_compile.sh
cd /home/zjason/rlvr-screens || exit 2
W=/home/zjason/orcd/pool/rlvr-ckpts/calibration/research_r93/d007s0_final/seeds/0/pretrained
ok=/tmp/r135ok.$SLURM_JOB_ID
rm -f $ok
( stat /home/zjason/orcd/pool/vllm-venv/bin/activate >/dev/null && /home/zjason/orcd/pool/vllm-venv/bin/python -c pass \
  && stat $W/model.safetensors >/dev/null && touch $ok ) &
for i in $(seq 1 60); do [ -e $ok ] && break; sleep 5; done
[ -e $ok ] || { echo "=== pool venv or weights unreadable in 300 s on $(hostname); exiting"; exit 3; }
rm -f $ok
echo "=== r135 on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) $(date -u +%FT%TZ)"
lib=$(g++ -print-file-name=libstdc++.so.6)
[[ $lib == /* ]] && export LD_LIBRARY_PATH=$(dirname $lib)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
source /home/zjason/orcd/pool/vllm-venv/bin/activate
module load cuda/13.0.1
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$HOME/.cache/huggingface \
  HF_HUB_OFFLINE=1 PYTHONPATH=$PWD OMP_NUM_THREADS=8
O=calibration/research_r135/check
mkdir -p $O
echo "=== r135 check $(date -u +%FT%TZ)"
python scripts/research/check_capture_compile.py --model $W --out $O/check.json > $O/stdout.log 2>&1
echo "rc=$? $(/usr/bin/grep -E 'rule_met|speedup' $O/stdout.log | tr -d '\n' | cut -c1-160)"
for arm in compiled; do
  O=calibration/research_r135/${arm}/seeds/0
  mkdir -p $O/run
  echo "=== r135 $arm $(date -u +%FT%TZ)"
  python scripts/research/profile_update.py --config calibration/research_r106/final_native_ctl/config.json --model $W \
    --out $O/profile.json --warm 300 --steps 2 --parts --out-dir $O/run --set max_steps=1000000 \
    --set update_host_index=true --set update_capture=true --set update_capture_width_step=16 \
    --set update_capture_row_step=8 --set update_capture_group_step=8 --set grammar_fast_path=true \
    --set fast_puzzle_draw=true --set prefetch_puzzle_draw=true --set compile_update=true > $O/stdout.log 2>&1
  echo "rc=$? $(tail -n 1 $O/stdout.log | cut -c1-160)"
done
echo "=== r135 done $(date -u +%FT%TZ)"
