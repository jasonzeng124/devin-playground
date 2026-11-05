cd ~/rlvr-screens; date -u
ls ~/orcd/pool/rlvr-ckpts/calibration/research_r46/pt80/seeds/0/pretrained ~/orcd/pool/rlvr-ckpts/calibration/research_r46/pt80/seeds/1/pretrained | tr '\n' ' '; echo
ls calibration/research_r46/pt80_rl17000/config.json
flock queue/lock bash -c 'printf "%s\n" "calibration/research_r46/pt80_rl17000 0" "calibration/research_r46/pt80_rl17000 1" | cat - queue/box_d.txt > queue/box_d.txt.tmp && mv queue/box_d.txt.tmp queue/box_d.txt; head -4 queue/box_d.txt'
