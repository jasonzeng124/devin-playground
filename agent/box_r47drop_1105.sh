cd ~/rlvr-screens; date -u
flock queue/lock bash -c 'grep -v -x -e "calibration/research_r47/a005_lr1.5e-6 2" queue/box_a.txt > queue/box_a.txt.tmp; mv queue/box_a.txt.tmp queue/box_a.txt; cat queue/box_a.txt'
