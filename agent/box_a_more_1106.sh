set -u
cd ~/rlvr-screens; date -u
( flock 9
  { cat queue/box_a.txt
    for s in 8 9; do echo "calibration/research_r39/a005_ctl $s"; echo "calibration/research_r40/a005_ent05 $s"; done
  } > queue/box_a.txt.tmp && mv queue/box_a.txt.tmp queue/box_a.txt
) 9>queue/lock
echo "box_a:"; cat queue/box_a.txt
