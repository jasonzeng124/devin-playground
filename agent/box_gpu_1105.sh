cd ~/rlvr-screens; date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for p in $(pgrep -u zjason -f "rlvr_speedrun.grpo|pretrain_scratch.py train"); do echo "$p $(ps -o etime= -p $p) $(tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep CUDA_VISIBLE) $(tr "\0" " " < /proc/$p/cmdline | grep -oE "research_r[0-9]+/[^/ ]+/config.json|--seed [0-9]+|g[0-9]-1" | tr "\n" " ")"; done
echo QUEUE; cat queue/box_d.txt; tail -6 queue/done.txt
