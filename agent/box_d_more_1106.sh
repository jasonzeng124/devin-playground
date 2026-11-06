set -u
cd ~/rlvr-screens; date -u
( flock 9
  { cat queue/box_d.txt
    echo "calibration/research_r36/d004rl_shape05_lr6e-6_t12_s15k 5"; echo "calibration/research_r44/d004rl_shape05_lr6e-6_s15k_box 5"
  } > queue/box_d.txt.tmp && mv queue/box_d.txt.tmp queue/box_d.txt
) 9>queue/lock
echo "box_d:"; cat queue/box_d.txt
