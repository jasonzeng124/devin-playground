cd ~/rlvr-screens
cat > queue/box_d.add <<'Q'
calibration/research_r44/d004rl_shape05_lr6e-6_s15k_box 4
calibration/research_r36/d004rl_shape05_lr6e-6_t12_s15k 4
calibration/research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k 4
calibration/research_r36/d004rl_shape05_lr6e-6_t12_s15k 3
calibration/research_r43/d004rl_shape05_lr6e-6_t14_s15k 2
calibration/research_r43/d004rl_shape05_lr6e-6_t14_s15k 3
Q
flock queue/lock bash -c 'cat queue/box_d.add queue/box_d.txt > queue/box_d.txt.tmp && mv queue/box_d.txt.tmp queue/box_d.txt'
rm queue/box_d.add; cat queue/box_d.txt; echo ---; tail -8 queue/done.txt
for g in 4 5 6 7; do f=slurm_logs/box-ip-172-31-17-122-g$g-2.out; grep -o "step[\"' :]*[0-9]*" calibration/research_r4*/*/seeds/*/train_log.jsonl 2>/dev/null | tail -0; done
for d in calibration/research_r44/d004rl_shape05_lr6e-6_s15k_box/seeds/3 calibration/research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k/seeds/3 calibration/research_r37/d004rl_shape05_lr6e-6_ent05_s15k/seeds/3 calibration/research_r39/a005_ctl/seeds/0; do echo "$d $(tail -1 $d/train_log.jsonl 2>/dev/null | cut -c1-120)"; done
