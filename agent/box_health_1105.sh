cd ~/rlvr-screens; date -u +%FT%TZ; wc -l queue/box_a.txt queue/box_d.txt
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for f in slurm_logs/box-*.out; do echo "== $f"; grep -E 'start calibration|done|FAIL|rror|Traceback' "$f" | tail -2; tail -1 "$f" | cut -c1-200; done
