cat /etc/rlvr-box.end; date -u +%s
cd ~/rlvr-screens && mkdir -p slurm_logs && wc -l queue/box_a.txt queue/box_d.txt && ls ~/orcd/pool/rlvr-ckpts/records/v3/track_d/004_dm_math_fp32_h200_warm/seeds/
SCREEN_RESUME=1 RESUME_EVERY_S=1200 RESERVE_MIN=30 QUEUE=box_a GPUS="0 1 2 3" PER_GPU=1 RUNNER=scripts/research/run_item.sh setsid nohup bash scripts/aws/box_workers.sh > slurm_logs/box_workers_a.out 2>&1 < /dev/null &
SCREEN_RESUME=1 RESUME_EVERY_S=600 RESERVE_MIN=30 QUEUE=box_d GPUS="4 5 6 7" PER_GPU=3 FIRST=1 RUNNER=scripts/research/run_item.sh setsid nohup bash scripts/aws/box_workers.sh > slurm_logs/box_workers_d.out 2>&1 < /dev/null &
sleep 20; cat slurm_logs/box_workers_a.out slurm_logs/box_workers_d.out
