cd ~/rlvr-screens; date -u
ls ~/orcd/pool/rlvr-ckpts/calibration/research_r46/pt150/seeds/0/pretrained | tr '\n' ' '; echo
ls calibration/research_r46/pt150_rl12500/config.json
flock queue/lock bash -c 'printf "%s\n" "calibration/research_r46/pt150_rl12500 0" | cat - queue/box_d.txt > queue/box_d.txt.tmp && mv queue/box_d.txt.tmp queue/box_d.txt; head -3 queue/box_d.txt'
