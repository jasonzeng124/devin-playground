set -u
cd ~/rlvr-screens
P=57292
A=calibration/research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k
cmd=$(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null)
echo "cmdline: $cmd"
case "$cmd" in
  *research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k/config.json*"--seed 4"*) ;;
  *) echo "PID $P is not r38 s4; abort"; exit 1;;
esac
(
  flock 9
  grep -vxF "$A 3" queue/box_d.txt > queue/box_d.txt.tmp; mv queue/box_d.txt.tmp queue/box_d.txt
  kill $P
) 9>queue/lock
sleep 20
kill -0 $P 2>/dev/null && echo "still alive" || echo "killed"
for s in 3 4; do
  r=$HOME/orcd/pool/rlvr-ckpts/$A/resume/$s
  ls -la "$r" 2>/dev/null
  rm -f "$r/state.pt" "$r/state.tmp" "$r"/*.jsonl "$r"/*.jsonl.tmp; rmdir "$r" 2>/dev/null || true
done
echo QUEUE; cat queue/box_d.txt
