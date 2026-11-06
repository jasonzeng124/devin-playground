set -u
cd ~/rlvr-screens; date -u
ok=1
for pa in "156971 research_r47/a005_lr1.5e-6/config.json" "162778 research_r48/a005_cos400/config.json"; do
  set -- $pa
  cmd=$(tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null)
  echo "$1: ${cmd:0:220}"
  case "$cmd" in *"$2"*"--seed 1"*) ;; *) echo "PID $1 mismatch"; ok=0;; esac
done
(
  flock 9
  { grep -vxF "calibration/research_r48/a005_cos400 2" queue/box_a.txt
    for s in 5 6 7; do echo "calibration/research_r39/a005_ctl $s"; echo "calibration/research_r40/a005_ent05 $s"; done
  } > queue/box_a.txt.tmp && mv queue/box_a.txt.tmp queue/box_a.txt
  [ $ok = 1 ] && kill 156971 162778
) 9>queue/lock
sleep 20
for p in 156971 162778; do kill -0 $p 2>/dev/null && echo "$p alive" || echo "$p gone"; done
for r in $HOME/orcd/pool/rlvr-ckpts/calibration/research_r47/a005_lr1.5e-6/resume/1 $HOME/orcd/pool/rlvr-ckpts/calibration/research_r48/a005_cos400/resume/1; do
  ls "$r" 2>/dev/null; rm -f "$r/state.pt" "$r/state.tmp" "$r"/*.jsonl "$r"/*.jsonl.tmp; rmdir "$r" 2>/dev/null || true
done
echo QUEUE; cat queue/box_a.txt
sleep 30; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4
