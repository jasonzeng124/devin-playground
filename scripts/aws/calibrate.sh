#!/usr/bin/env bash
# Unattended plateau calibration on one AWS GPU instance (run from the repo root):
#
#   bash scripts/aws/calibrate.sh
#
# 1. retries run-instances across availability zones until capacity appears (MAX_WAIT_MIN);
#    after FALLBACK_AFTER_MIN it also accepts FALLBACK_TYPE (any GPU works: calibration uses steps)
# 2. copies the repo, starts scripts/aws/remote_calibrate.sh under nohup
# 3. every SYNC_MIN minutes rsyncs calibration/ results + status back (logs survive a crash)
# 4. when the remote writes ~/DONE (or the instance disappears), syncs a last time and
#    terminates the instance. The instance also powers off -> terminates on its own after
#    BACKSTOP_MIN minutes, so a dead local machine cannot leave it running.
set -uo pipefail

REGION=${REGION:-us-east-1}
AZS=${AZS:-"us-east-1a us-east-1b us-east-1c us-east-1d"}
INSTANCE_TYPE=${INSTANCE_TYPE:-g6e.2xlarge}   # L40S 48 GB
FALLBACK_TYPE=${FALLBACK_TYPE:-g5.2xlarge}    # A10G 24 GB
FALLBACK_AFTER_MIN=${FALLBACK_AFTER_MIN:-30}
AMI=${AMI:-ami-012ba162b9cd2729c}  # Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)
KEY=${KEY:-rlvr-speedrun}
SG=${SG:-sg-0160538477117c054}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-360}
SYNC_MIN=${SYNC_MIN:-10}
BACKSTOP_MIN=${BACKSTOP_MIN:-840}
LOG=${LOG:-results/aws_calibrate.log}
SSH=(ssh -i ~/.ssh/$KEY.pem -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30)
mkdir -p results
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

userdata=$(mktemp)
printf '#!/bin/bash\nshutdown -h +%s\n' "$BACKSTOP_MIN" >"$userdata"

iid=""
start_s=$(date +%s)
deadline=$(( start_s + MAX_WAIT_MIN * 60 ))
while [[ -z "$iid" && $(date +%s) -lt $deadline ]]; do
  types=$INSTANCE_TYPE
  (( $(date +%s) - start_s >= FALLBACK_AFTER_MIN * 60 )) && types="$INSTANCE_TYPE $FALLBACK_TYPE"
  for type in $types; do for az in $AZS; do
    subnet=$(aws ec2 describe-subnets --region "$REGION" --filters Name=availability-zone,Values="$az" Name=default-for-az,Values=true --query 'Subnets[0].SubnetId' --output text)
    iid=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$type" --key-name "$KEY" \
      --subnet-id "$subnet" --security-group-ids "$SG" --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
      --user-data "file://$userdata" --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=rlvr-calibration}]' \
      --query 'Instances[0].InstanceId' --output text 2>>"$LOG") && { log "launched $iid ($type) in $az"; break 2; }
    iid=""
  done; done
  [[ -z "$iid" ]] && { log "no capacity for [$types] in $AZS; retrying in 5 min"; sleep 300; }
done
[[ -z "$iid" ]] && { log "gave up after $MAX_WAIT_MIN min without capacity"; exit 1; }
echo "$iid" >results/aws_instance_id

aws ec2 wait instance-running --region "$REGION" --instance-ids "$iid"
ip=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
host=ubuntu@$ip
log "running at $ip; waiting for ssh"
until "${SSH[@]}" "$host" true 2>/dev/null; do sleep 15; done

rsync -az -e "${SSH[*]}" --exclude .venv --exclude results --exclude '__pycache__' ./ "$host:rlvr/"
"${SSH[@]}" "$host" "nohup bash rlvr/scripts/aws/remote_calibrate.sh >/dev/null 2>&1 &"
log "remote job started"

sync_back() {
  rsync -az -e "${SSH[*]}" "$host:rlvr/calibration/" calibration/ 2>/dev/null
  rsync -az -e "${SSH[*]}" "$host:smoke/" results/aws_smoke/ 2>/dev/null
  rsync -az -e "${SSH[*]}" "$host:status.log" "$host:remote.log" "$host:compile_check.log" results/ 2>/dev/null
}
while true; do
  sleep $((SYNC_MIN * 60))
  state=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" --query 'Reservations[0].Instances[0].State.Name' --output text)
  if [[ "$state" != running ]]; then log "instance is $state (backstop?); stopping"; break; fi
  sync_back
  log "synced; last status: $(tail -1 results/status.log 2>/dev/null)"
  if "${SSH[@]}" "$host" test -f DONE; then sync_back; log "remote finished"; break; fi
done
aws ec2 terminate-instances --region "$REGION" --instance-ids "$iid" >/dev/null && log "terminated $iid"
