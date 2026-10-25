#!/usr/bin/env bash
# Prepare one AWS capacity block as a box the Engaging research agent drives over ssh: a launch
# template that runs the instance inside the reservation, attached to a fixed network interface
# (so the box always has the same Elastic IP), with scripts/aws/box_userdata.sh as user data.
# With SCHEDULE=1 it also creates EventBridge Scheduler one-shots that call RunInstances with that
# template at the block start (+1, +10 and +25 min; once the reservation is full the later calls
# just fail), so the launch needs neither the laptop nor the agent. The instance runs until AWS
# ends the block. Run from the repo root on the laptop:
#
#   REGION=us-east-1 CR=cr-03fec7969fbda8633 ENI=eni-03f79eb9d98654069 SCHEDULE=1 bash scripts/aws/box_launch.sh
#   TEST=1 REGION=us-east-1 ENI=eni-03f79eb9d98654069 bash scripts/aws/box_launch.sh   # t3.micro, on demand, now+6 min
#
# Needs on Engaging: ~/.ssh/rlvr_aws(.pub) (the agent's key) and ~/rlvr-screens/agent/vllm-venv-freeze.txt.
set -euo pipefail
REGION=${REGION:?} ENI=${ENI:?}
export AWS_DEFAULT_REGION=$REGION
account=$(aws sts get-caller-identity --query Account --output text)
tmp=$(mktemp -d)
if [[ -n "${TEST:-}" ]]; then
  itype=t3.micro start=$(date -u -d "+6 min" +%FT%TZ) end=$(date -u -d "+2 hours" +%FT%TZ) name=rlvr-box-test
  market=''
else
  read -r itype start end < <(aws ec2 describe-capacity-reservations --capacity-reservation-ids "${CR:?}" \
    --query 'CapacityReservations[0].[InstanceType,StartDate,EndDate]' --output text)
  name=rlvr-box-$CR
  market='"InstanceMarketOptions":{"MarketType":"capacity-block"},"CapacityReservationSpecification":{"CapacityReservationTarget":{"CapacityReservationId":"'$CR'"}},'
fi
ami=${AMI:-$(aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)}
agent_key=$(ssh -o BatchMode=yes engaging cat .ssh/rlvr_aws.pub)
laptop_key=$(ssh-keygen -y -f ~/.ssh/rlvr-speedrun.pem)
freeze=$(ssh -o BatchMode=yes engaging cat rlvr-screens/agent/vllm-venv-freeze.txt | gzip -9 | base64 -w0)
sed -e "s|@END_EPOCH@|$(date -d "$end" +%s)|" -e "s|@AGENT_KEY@|$agent_key|" -e "s|@LAPTOP_KEY@|$laptop_key|" \
  -e "s|@FREEZE_GZ_B64@|$freeze|" scripts/aws/box_userdata.sh >"$tmp/userdata"
(( $(stat -c %s "$tmp/userdata") < 16000 )) || { echo "user data too big" >&2; exit 1; }
aws ec2 delete-launch-template --launch-template-name "$name" >/dev/null 2>&1 || true
lt=$(aws ec2 create-launch-template --launch-template-name "$name" --query LaunchTemplate.LaunchTemplateId --output text \
  --tag-specifications 'ResourceType=launch-template,Tags=[{Key=rlvr-box,Value=true},{Key=Project,Value=rlvr-speedrun}]' \
  --launch-template-data "$(cat <<JSON
{"ImageId":"$ami","InstanceType":"$itype",$market
 "NetworkInterfaces":[{"DeviceIndex":0,"NetworkInterfaceId":"$ENI"}],
 "UserData":"$(base64 -w0 "$tmp/userdata")",
 "BlockDeviceMappings":[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":${VOLUME_GB:-2000},"VolumeType":"gp3","Iops":6000,"Throughput":500,"DeleteOnTermination":true}}],
 "TagSpecifications":[{"ResourceType":"instance","Tags":[{"Key":"Name","Value":"$name"},{"Key":"Project","Value":"rlvr-speedrun"}]}]}
JSON
)")
ip=$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI" --query 'NetworkInterfaces[0].Association.PublicIp' --output text)
echo "$name: $itype, $start .. $end, launch template $lt ($ami), ssh zjason@$ip"
[[ -n "${TEST:-}${SCHEDULE:-}" ]] || exit 0

role=arn:aws:iam::$account:role/rlvr-scheduler
if ! aws iam get-role --role-name rlvr-scheduler >/dev/null 2>&1; then
  aws iam create-role --role-name rlvr-scheduler --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
fi
aws iam put-role-policy --role-name rlvr-scheduler --policy-name run-rlvr-box --policy-document "$(cat <<JSON
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":"ec2:RunInstances","Resource":"*","Condition":{"ArnLike":{"ec2:LaunchTemplate":"arn:aws:ec2:*:$account:launch-template/*"},"Bool":{"ec2:IsLaunchTemplateResource":"true"}}},
 {"Effect":"Allow","Action":"ec2:RunInstances","Resource":"arn:aws:ec2:*:$account:launch-template/*","Condition":{"StringEquals":{"ec2:ResourceTag/rlvr-box":"true"}}},
 {"Effect":"Allow","Action":"ec2:RunInstances","Resource":"arn:aws:ec2:*::image/*"},
 {"Effect":"Allow","Action":"ec2:CreateTags","Resource":"*","Condition":{"StringEquals":{"ec2:CreateAction":"RunInstances"}}}]}
JSON
)"
sleep 15  # a new role or policy is not usable right away
input=$(printf '{"LaunchTemplate":{"LaunchTemplateId":"%s"},"MinCount":1,"MaxCount":1}' "$lt")
for off in ${OFFSETS:-1 10 25}; do
  [[ -n "${TEST:-}" ]] && off=0
  at=$(date -u -d "$start + $off min" +%FT%T)
  aws scheduler delete-schedule --name "$name-$off" >/dev/null 2>&1 || true
  aws scheduler create-schedule --name "$name-$off" --schedule-expression "at($at)" --schedule-expression-timezone UTC \
    --flexible-time-window Mode=OFF --action-after-completion DELETE \
    --target "$(python3 -c 'import json, sys; print(json.dumps({"Arn": "arn:aws:scheduler:::aws-sdk:ec2:runInstances",
      "RoleArn": sys.argv[1], "Input": sys.argv[2], "RetryPolicy": {"MaximumRetryAttempts": 3, "MaximumEventAgeInSeconds": 600}}))' "$role" "$input")" >/dev/null
  echo "scheduled RunInstances at $at UTC ($name-$off)"
  [[ -n "${TEST:-}" ]] && break
done
