#!/bin/bash
# r132 (LOG 2026-11-25): where r131's step time goes outside t_sample and t_update, and the update's exact token
# count for its MFU. Two repeats (a, b) of r131's off arm (r106's final_native_ctl config, D 007 seed 0's final
# weights, capture with the fine steps, part timers on), which now also log t_part_draw (the puzzle draw before
# the step), t_part_sync (the rollout engine's weight sync), t_part_step (the step's timed span) and update_tokens.
# 300 timed updates an arm (about 0.43 s a step in r131); the read is the medians over steps 100-299.
# Submit from slurm_logs inside `bash -lc` (the login env's g++ for FlashInfer), e.g.
#   sbatch -p mit_preemptable --time=01:00:00 --gres=gpu:b200:1 -c 16 --mem=96G \
#     --exclude=node5500-c1,node5800-c1,node5702-c1 -J rlvr-prep \
#     -o /home/zjason/rlvr-screens/slurm_logs/r132_%j.out scripts/research/r132_loop_parts.sh
cd /home/zjason/rlvr-screens || exit 2
W=/home/zjason/orcd/pool/rlvr-ckpts/calibration/research_r93/d007s0_final/seeds/0/pretrained
ok=/tmp/r132ok.$SLURM_JOB_ID
rm -f $ok
( stat /home/zjason/orcd/pool/vllm-venv/bin/activate >/dev/null && /home/zjason/orcd/pool/vllm-venv/bin/python -c pass \
  && stat $W/model.safetensors >/dev/null && touch $ok ) &
for i in $(seq 1 60); do [ -e $ok ] && break; sleep 5; done
[ -e $ok ] || { echo "=== pool venv or weights unreadable in 300 s on $(hostname); exiting"; exit 3; }
rm -f $ok
echo "=== r132 on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) $(date -u +%FT%TZ)"
lib=$(g++ -print-file-name=libstdc++.so.6)
[[ $lib == /* ]] && export LD_LIBRARY_PATH=$(dirname $lib)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
source /home/zjason/orcd/pool/vllm-venv/bin/activate
module load cuda/13.0.1
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$HOME/.cache/huggingface \
  HF_HUB_OFFLINE=1 PYTHONPATH=$PWD OMP_NUM_THREADS=8
for arm in a b; do
  O=calibration/research_r132/${arm}/seeds/0
  mkdir -p $O/run
  echo "=== r132 $arm $(date -u +%FT%TZ)"
  python scripts/research/profile_update.py --config calibration/research_r106/final_native_ctl/config.json --model $W \
    --out $O/profile.json --warm 300 --steps 2 --parts --out-dir $O/run --set max_steps=1000000 \
    --set update_host_index=true --set update_capture=true --set update_capture_width_step=16 \
    --set update_capture_row_step=8 --set update_capture_group_step=8 --set grammar_fast_path=false > $O/stdout.log 2>&1
  echo "rc=$? $(tail -n 1 $O/stdout.log | cut -c1-160)"
done
echo "=== r132 done $(date -u +%FT%TZ)"
