#!/usr/bin/env bash
# Pull a spot run's results from S3 into the repo and show each seed's progress.
#
#   bash scripts/aws/spot_fetch.sh <RUN>        # RUN as printed by spot_launch.sh
set -euo pipefail
RUN=${1:?usage: spot_fetch.sh <RUN>}
REGION=${REGION:-us-east-1}
RECORD=${RECORD:-calibration/plateau_005}
BUCKET=${BUCKET:-rlvr-speedrun-$(aws sts get-caller-identity --query Account --output text)}
export AWS_DEFAULT_REGION=$REGION

for prefix in $(aws s3 ls "s3://$BUCKET/runs/$RUN/" | awk '/PRE seed_/ {print $2}'); do
  aws s3 sync --quiet "s3://$BUCKET/runs/$RUN/${prefix}seeds/" "$RECORD/seeds/"
  aws s3 ls "s3://$BUCKET/runs/$RUN/${prefix}extra/" >/dev/null 2>&1 && aws s3 sync --quiet "s3://$BUCKET/runs/$RUN/${prefix}extra/" "$RECORD/extra/"
  mkdir -p "results/$RUN" && aws s3 cp --quiet "s3://$BUCKET/runs/$RUN/${prefix}status.log" "results/$RUN/${prefix%/}.status.log" || true
  echo "${prefix%/}: $(tail -1 "results/$RUN/${prefix%/}.status.log" 2>/dev/null)"
done
python3 - "$RECORD" <<'EOF'
import json, pathlib, sys
seeds = pathlib.Path(sys.argv[1], "seeds")
for seed in sorted(seeds.iterdir()) if seeds.is_dir() else []:
    evals = [json.loads(l) for l in (seed / "eval_log.jsonl").open()] if (seed / "eval_log.jsonl").exists() else []
    train = [json.loads(l) for l in (seed / "train_log.jsonl").open()] if (seed / "train_log.jsonl").exists() else []
    mem = [r["gpu_mem_gb"] for r in train if "gpu_mem_gb" in r]
    curve = " ".join(f"{e['step']}:{e['pass_rate']:.3f}" for e in evals[-6:])
    print(f"seed {seed.name}: step {train[-1]['step'] if train else 0}, gpu mem {mem[0]:.0f}->{mem[-1]:.0f} GB" if mem else f"seed {seed.name}: step {train[-1]['step'] if train else 0}", "|", curve)
EOF
echo "instances: $(aws ec2 describe-instances --filters Name=tag:Project,Values=rlvr-speedrun Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value,State.Name]' --output text | tr '\n' ' ')"
echo "fleets: $(aws ec2 describe-fleets --filters Name=fleet-state,Values=submitted,active,modifying --query 'Fleets[?Tags[?Key==`Project`&&Value==`rlvr-speedrun`]].[FleetId,FleetState,FulfilledCapacity]' --output text | tr '\n' ' ')"
