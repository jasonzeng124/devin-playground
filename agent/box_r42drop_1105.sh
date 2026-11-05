cd ~/rlvr-screens; date -u
flock queue/lock bash -c 'grep -v -x -e "calibration/research_r42/a005_lr7e-7 3" -e "calibration/research_r42/a005_lr7e-7 4" queue/box_a.txt | cat - <(printf "%s\n" "calibration/research_r47/a005_lr1.5e-6 2" "calibration/research_r48/a005_cos400 2") > queue/box_a.txt.tmp && mv queue/box_a.txt.tmp queue/box_a.txt; cat queue/box_a.txt'
