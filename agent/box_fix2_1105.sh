cd ~/rlvr-screens; date -u
flock queue/lock bash -c '
P=$(pgrep -u zjason -f "rlvr_speedrun.grpo --config calibration/research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k/config.json --device cuda --seed 3 ")
echo "PIDS=$P"; [ $(echo $P | wc -w) = 1 ] || { echo "not exactly one pid; abort"; exit 1; }
ps -o pid,etime,args -p $P | cut -c1-140; tr "\0" "\n" < /proc/$P/environ | grep CUDA_VISIBLE
kill $P; for i in $(seq 30); do ps -p $P >/dev/null || break; sleep 2; done; ps -p $P >/dev/null && echo STILL_ALIVE
R=~/orcd/pool/rlvr-ckpts/calibration/research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k/resume/3
ls -la $R; rm -f $R/state.pt $R/train_log.jsonl $R/eval_log.jsonl; rmdir $R
echo "calibration/research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k 3" >> queue/box_d.txt.tmp0
cat queue/box_d.txt queue/box_d.txt.tmp0 > queue/box_d.txt.tmp && mv queue/box_d.txt.tmp queue/box_d.txt && rm queue/box_d.txt.tmp0
echo QUEUE:; cat queue/box_d.txt'
sleep 40; tail -2 queue/done.txt
for p in $(pgrep -u zjason -f "rlvr_speedrun.grpo"); do echo "$p $(tr "\0" "\n" < /proc/$p/environ | grep CUDA_VISIBLE) $(tr "\0" " " < /proc/$p/cmdline | grep -oE "research_r[0-9]+/[^/]+|--seed [0-9]+" | tr "\n" " ")"; done
