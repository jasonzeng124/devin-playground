cd ~/rlvr-screens; date -u
flock queue/lock bash -c '
ps -o pid,args -p 23418,17651 | cut -c1-160
kill 23418 17651
for i in $(seq 30); do ps -p 23418,17651 >/dev/null || break; sleep 2; done
ps -p 23418,17651 >/dev/null && echo STILL_ALIVE
C=~/orcd/pool/rlvr-ckpts/calibration
for p in research_r37/d004rl_shape05_lr6e-6_ent05_s15k/resume/4 research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k/resume/3; do
  ls -la $C/$p; rm -f $C/$p/state.pt $C/$p/train_log.jsonl $C/$p/eval_log.jsonl; rmdir $C/$p; done
printf "%s\n" "calibration/research_r37/d004rl_shape05_lr6e-6_ent05_s15k 4" "calibration/research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k 3" > queue/box_d.add2
cat queue/box_d.add2 queue/box_d.txt > queue/box_d.txt.tmp && mv queue/box_d.txt.tmp queue/box_d.txt
echo QUEUE:; cat queue/box_d.txt'
sleep 45; tail -4 queue/done.txt; pgrep -u zjason -af "screen_seed.sh" | cut -c1-120
