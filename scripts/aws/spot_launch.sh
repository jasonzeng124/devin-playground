#!/usr/bin/env bash
# Queue one single-instance spot EC2 Fleet ("maintain") per seed. AWS launches each instance
# whenever capacity appears in any zone (until VALID_HOURS), and replaces it if spot reclaims
# it; the instance trains its seed unattended, pushes results to S3, then deletes its fleet
# and terminates (scripts/aws/spot_job.sh). Run from the repo root; the laptop can sleep or
# change IP afterwards. Fetch results with scripts/aws/spot_fetch.sh <RUN>.
#
#   bash scripts/aws/spot_launch.sh [SEED...]     # default seeds: 0 1 2 3
#
# Creates on first use (idempotent): S3 bucket $BUCKET and IAM role/instance profile
# rlvr-runner (S3 access to that bucket; may delete EC2 fleets and terminate instances
# tagged Project=rlvr-speedrun).
#
# Benchmark v3 records (scripts/run_seed_v3.sh, the reference software in a fresh env):
#   V3=1 AMI=ami-0c244e62d32d4ca4e RECORD=records/v3/track_d/000_baseline bash scripts/aws/spot_launch.sh 0 1 2 3 4
# Several seeds on one 8x H100 instance (one per GPU; the argument then just names the box):
#   V3=1 INSTANCE_TYPE=p5.48xlarge SLOTS="records/v3/track_a/000_baseline:0 records/v3/track_d/000_baseline:0" ... spot_launch.sh b0
# Dev-only screens (no test ladder): V3=1 STAGE=train RECORD=calibration/<batch>/<arm> ...
set -euo pipefail

REGION=${REGION:-us-east-1}
AZS=${AZS:-"us-east-1a us-east-1b us-east-1c us-east-1d us-east-1e us-east-1f"}
INSTANCE_TYPE=${INSTANCE_TYPE:-p5.4xlarge}   # 1x H100 80GB
AMI=${AMI:-ami-012ba162b9cd2729c}            # Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)
RECORD=${RECORD:-calibration/plateau_005}
EXTRA_FLAGS=${EXTRA_FLAGS:-}
BACKSTOP_MIN=${BACKSTOP_MIN:-600}   # per instance
VALID_HOURS=${VALID_HOURS:-48}       # how long a fleet keeps waiting for capacity
CAPACITY=${CAPACITY:-spot}             # or on-demand (same fleet, no spot interruptions, ~2x the price)
VOLUME_GB=${VOLUME_GB:-200}
SG=${SG:-sg-0160538477117c054}
KEY=${KEY-rlvr-speedrun}  # empty: no SSH key (the job needs none)
account=$(aws sts get-caller-identity --query Account --output text)
BUCKET=${BUCKET:-rlvr-speedrun-$account}
RUN=${RUN:-$(basename "$RECORD")-$(date -u +%Y%m%d-%H%M)}
seeds=${*:-0 1 2 3}
export AWS_DEFAULT_REGION=$REGION

aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null || aws s3api create-bucket --bucket "$BUCKET" >/dev/null
new_profile=0
if ! aws iam get-instance-profile --instance-profile-name rlvr-runner >/dev/null 2>&1; then
  aws iam create-role --role-name rlvr-runner --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam create-instance-profile --instance-profile-name rlvr-runner >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name rlvr-runner --role-name rlvr-runner
  new_profile=1
fi
aws iam put-role-policy --role-name rlvr-runner --policy-name rlvr-runner --policy-document "$(cat <<EOF
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket"],"Resource":["arn:aws:s3:::$BUCKET","arn:aws:s3:::$BUCKET/*"]},
 {"Effect":"Allow","Action":["ec2:DescribeInstances","ec2:DeleteFleets","ec2:DeleteLaunchTemplate"],"Resource":"*"},
 {"Effect":"Allow","Action":"ec2:TerminateInstances","Resource":"*","Condition":{"StringEquals":{"ec2:ResourceTag/Project":"rlvr-speedrun"}}}]}
EOF
)"
(( new_profile )) && sleep 20  # a new instance profile is not usable right away

tarball=$(mktemp --suffix .tgz)
{ git ls-files -co --exclude-standard | grep -v "^$RECORD/\(seeds\|extra\)/\|^results/\|^calibration/\|^records/.*/seeds/"
  git ls-files -co --exclude-standard "$RECORD" | grep -v "^$RECORD/\(seeds\|extra\)/"; } | sort -u | tar czf "$tarball" -T -
aws s3 cp --quiet "$tarball" "s3://$BUCKET/runs/$RUN/code.tgz"
echo "run $RUN: code in s3://$BUCKET/runs/$RUN/"

for seed in $seeds; do
  userdata=$(mktemp)
  cat >"$userdata" <<EOF
#!/bin/bash
cat >/etc/rlvr.env <<ENV
BUCKET=$BUCKET
RUN=$RUN
SEED=$seed
RECORD=$RECORD
EXTRA_FLAGS="$EXTRA_FLAGS"
V3=${V3:-0}
SLOTS="${SLOTS:-}"
AWS_DEFAULT_REGION=$REGION
CONFIRM_LIMIT=${CONFIRM_LIMIT:-}
STAGE=${STAGE:-all}
DEADLINE=\$(( \$(date +%s) + $BACKSTOP_MIN * 60 ))
ENV
mkdir -p /opt/rlvr && aws --region $REGION s3 cp s3://$BUCKET/runs/$RUN/code.tgz /tmp/code.tgz && tar xzf /tmp/code.tgz -C /opt/rlvr
cat >/etc/systemd/system/rlvr.service <<UNIT
[Unit]
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
Environment=HOME=/root
ExecStart=/bin/bash /opt/rlvr/scripts/aws/spot_job.sh
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload && systemctl enable --now rlvr.service
EOF
  overrides=$(for az in $AZS; do
    aws ec2 describe-subnets --filters Name=availability-zone,Values="$az" Name=default-for-az,Values=true --query 'Subnets[0].SubnetId' --output text
  done | grep '^subnet-' | sed 's/.*/{"SubnetId":"&"}/' | paste -sd,)
  lt=$(aws ec2 create-launch-template --launch-template-name "rlvr-$RUN-s$seed" --query 'LaunchTemplate.LaunchTemplateId' --output text \
    --launch-template-data "$(cat <<JSON
{"ImageId":"$AMI","InstanceType":"$INSTANCE_TYPE",$([[ -n $KEY ]] && echo "\"KeyName\":\"$KEY\",")"SecurityGroupIds":["$SG"],
 "IamInstanceProfile":{"Name":"rlvr-runner"},"UserData":"$(base64 -w0 "$userdata")",
 "BlockDeviceMappings":[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":$VOLUME_GB,"VolumeType":"gp3","DeleteOnTermination":true}}],
 "TagSpecifications":[{"ResourceType":"instance","Tags":[{"Key":"Name","Value":"rlvr-$RUN-s$seed"},{"Key":"Project","Value":"rlvr-speedrun"}]}]}
JSON
)")
  fleet=$(aws ec2 create-fleet --type maintain --valid-until "$(date -u -d "+$VALID_HOURS hours" +%FT%TZ)" \
    --no-terminate-instances-with-expiration \
    --target-capacity-specification TotalTargetCapacity=1,DefaultTargetCapacityType=$CAPACITY \
    $([[ $CAPACITY == spot ]] && echo --spot-options AllocationStrategy=price-capacity-optimized,InstanceInterruptionBehavior=terminate) \
    --launch-template-configs "[{\"LaunchTemplateSpecification\":{\"LaunchTemplateId\":\"$lt\",\"Version\":\"\$Latest\"},\"Overrides\":[$overrides]}]" \
    --tag-specifications "ResourceType=fleet,Tags=[{Key=Name,Value=rlvr-$RUN-s$seed},{Key=Project,Value=rlvr-speedrun}]" \
    --query FleetId --output text)
  echo "seed $seed: queued $fleet ($INSTANCE_TYPE, waits up to $VALID_HOURS h for capacity)"
done
