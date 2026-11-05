cd ~/rlvr-screens; date -u
flock queue/lock bash -c '
P=42956; ps -o pid,etime,args -p $P | cut -c1-140
tr "\0" " " < /proc/$P/cmdline | grep -q "research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k/config.json --device cuda --seed 3 " || { echo "pid mismatch; abort"; exit 1; }
kill $P; for i in $(seq 30); do ps -p $P >/dev/null || break; sleep 2; done; ps -p $P >/dev/null && echo STILL_ALIVE
R=~/orcd/pool/rlvr-ckpts/calibration/research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k/resume/3
ls -la $R; rm -f $R/state.pt $R/train_log.jsonl $R/eval_log.jsonl; rmdir $R
cp queue/box_d.txt queue/box_d.txt.tmp && echo "calibration/research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k 3" >> queue/box_d.txt.tmp && mv queue/box_d.txt.tmp queue/box_d.txt
echo QUEUE:; cat queue/box_d.txt'
sleep 40; tail -2 queue/done.txt
for p in $(pgrep -u zjason -f "rlvr_speedrun.grpo"); do echo "$p $(tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep CUDA_VISIBLE) $(tr "\0" " " < /proc/$p/cmdline | grep -oE "research_r[0-9]+/[^/ ]+/config.json|--seed [0-9]+" | tr "\n" " ")"; done
