cd ~/rlvr-screens; date -u
pgrep -u zjason -af 'research_r37/d004rl_shape05_lr6e-6_ent05_s15k|research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k' | cut -c1-250
ps -u zjason -o pid,ppid,etime,args --forest | grep -E 'run_item|python' | grep -v grep | cut -c1-200 | head -40
for p in calibration/research_r37/d004rl_shape05_lr6e-6_ent05_s15k/resume/4 calibration/research_r38/d004rl_shape05_lr6e-6_t12_ent05_s15k/resume/3; do echo "$p:"; ls -la ~/orcd/pool/rlvr-ckpts/$p 2>&1; done
grep -n "SCREEN_RESUME\|resume" scripts/research/run_item.sh | head -20
