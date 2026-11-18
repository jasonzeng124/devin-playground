#!/usr/bin/env bash
# AWS side of a self-running box: the launch template that makes an instance fetch its work and start by itself
# (user data: scripts/aws/box_userdata.sh with the self-running stage on; on the box: scripts/aws/box_boot.sh), the
# schedules that launch it with nobody awake, and the roles both need. Run on Engaging from the repo root. What the
# box fetches is staged with scripts/aws/box_s3.sh. <box> is a capacity reservation id (cr-...) or "test".
#
#   bash scripts/aws/box_auto.sh iam                  roles: rlvr-scheduler may launch from a template tagged
#                                                     rlvr-box=true; rlvr-box (instance profile) may read and write
#                                                     the bucket's box/ and box-state/ prefixes
#   bash scripts/aws/box_auto.sh lifecycle            bucket rules: box-state/ expires after 7 days, box/ after 90
#   bash scripts/aws/box_auto.sh template <box>       create the launch template rlvr-box-<box>, or add a version to
#                                                     it and make that the default (older versions stay)
#   bash scripts/aws/box_auto.sh schedule <box> [min ...]   one-time schedules that call RunInstances with the
#                                                     template's default version, <min> minutes after the block
#                                                     starts (test: after now). Later ones relaunch a box that died;
#                                                     while an instance is up they are refused (its network interface
#                                                     is in use and the block holds one instance)
#   bash scripts/aws/box_auto.sh unschedule <box>     delete its schedules
#   bash scripts/aws/box_auto.sh dryrun <box>         RunInstances --dry-run under the rlvr-scheduler role
#   bash scripts/aws/box_auto.sh show <box>           the reservation, the template's default version, schedules,
#                                                     instances, and RunInstances calls of the last hours
#
# The ssh keys and the venv pin list are copied from the user data of an existing box template (FROM, default the
# 11-28 one), so a new template builds the same venv as the boxes that have run. ENI and AMI override the network
# interface (default: the one described "rlvr box <zone>") and the image (default: FROM's).
# Do not run scripts/aws/box_launch.sh for a reservation set up here: it deletes the template and makes one without
# the self-running stage. If that happened, run `template` again (the schedules name the template, not a version).
set -euo pipefail
cmd=${1:?usage: box_auto.sh iam|lifecycle|template|schedule|unschedule|dryrun|show [<box>] ...}
shift
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
account=$(aws sts get-caller-identity --query Account --output text)
BUCKET=${BUCKET:-rlvr-speedrun-$account}
FROM=${FROM:-rlvr-box-cr-08927241900b5884c}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

box() {  # sets name cr itype start end zone for <box>
  local b=${1:?box: a capacity reservation id or "test"}
  if [[ $b == test ]]; then
    name=rlvr-box-test cr='' itype=${ITYPE:-t3.large} zone=${ZONE:-us-east-1d}
    start=$(date -u +%FT%TZ) end=$(date -u -d "+3 hours" +%FT%TZ)
  else
    name=rlvr-box-$b cr=$b
    read -r itype start end zone < <(aws ec2 describe-capacity-reservations --capacity-reservation-ids "$b" \
      --query 'CapacityReservations[0].[InstanceType,StartDate,EndDate,AvailabilityZone]' --output text)
  fi
}
default_version() {  # <template name> <file>: the template's default version as JSON; fails if there is no template
  aws ec2 describe-launch-template-versions --launch-template-name "$1" --versions '$Default' \
    --query 'LaunchTemplateVersions[0]' >"$2" 2>/dev/null && [[ $(jq -r '.LaunchTemplateName // empty' "$2") == "$1" ]]
}

case $cmd in
  iam)
    # the launcher. The 11-05 policy let RunInstances touch only resources named in the template, and the security
    # group and subnet come from the network interface, not from the template: every scheduled launch was refused.
    aws iam get-role --role-name rlvr-scheduler >/dev/null 2>&1 || aws iam create-role --role-name rlvr-scheduler \
      --assume-role-policy-document '{"Version":"2012-10-17","Statement":[]}' >/dev/null
    # the agent's user may take the role too: that is how `dryrun` tests the policy
    aws iam update-assume-role-policy --role-name rlvr-scheduler --policy-document "$(jq -n --arg a "$account" '
      {Version: "2012-10-17", Statement: [
        {Effect: "Allow", Principal: {Service: "scheduler.amazonaws.com"}, Action: "sts:AssumeRole"},
        {Effect: "Allow", Principal: {AWS: "arn:aws:iam::\($a):user/rlvr-agent"}, Action: "sts:AssumeRole"}]}')"
    aws iam put-role-policy --role-name rlvr-scheduler --policy-name run-rlvr-box --policy-document "$(jq -n --arg a "$account" '
      {Version: "2012-10-17", Statement: [
        {Effect: "Allow", Action: "ec2:RunInstances", Resource: "arn:aws:ec2:*:\($a):launch-template/*",
         Condition: {StringEquals: {"ec2:ResourceTag/rlvr-box": "true"}}},
        {Effect: "Allow", Action: "ec2:RunInstances", Resource: "arn:aws:ec2:*:\($a):instance/*",
         Condition: {ArnLike: {"ec2:LaunchTemplate": "arn:aws:ec2:*:\($a):launch-template/*"}}},
        {Effect: "Allow", Action: "ec2:RunInstances", Resource: [
          "arn:aws:ec2:*:\($a):subnet/*", "arn:aws:ec2:*:\($a):security-group/*", "arn:aws:ec2:*:\($a):network-interface/*",
          "arn:aws:ec2:*:\($a):volume/*", "arn:aws:ec2:*:\($a):key-pair/*", "arn:aws:ec2:*:\($a):capacity-reservation/*",
          "arn:aws:ec2:*::image/*", "arn:aws:ec2:*::snapshot/*"]},
        {Effect: "Allow", Action: "ec2:CreateTags", Resource: "*", Condition: {StringEquals: {"ec2:CreateAction": "RunInstances"}}},
        {Effect: "Allow", Action: "iam:PassRole", Resource: "arn:aws:iam::\($a):role/rlvr-box",
         Condition: {StringEquals: {"iam:PassedToService": "ec2.amazonaws.com"}}}]}')"
    # the box itself: reads box/ and box-state/, writes box/<name>/out/ and box-state/, deletes only in box-state/
    aws iam get-role --role-name rlvr-box >/dev/null 2>&1 || aws iam create-role --role-name rlvr-box \
      --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    aws iam put-role-policy --role-name rlvr-box --policy-name box-s3 --policy-document "$(jq -n --arg b "arn:aws:s3:::$BUCKET" '
      {Version: "2012-10-17", Statement: [
        {Effect: "Allow", Action: "s3:ListBucket", Resource: $b, Condition: {StringLike: {"s3:prefix": ["box/*", "box-state/*"]}}},
        {Effect: "Allow", Action: "s3:GetObject", Resource: ["\($b)/box/*", "\($b)/box-state/*"]},
        {Effect: "Allow", Action: ["s3:PutObject", "s3:AbortMultipartUpload"], Resource: ["\($b)/box/*/out/*", "\($b)/box-state/*"]},
        {Effect: "Allow", Action: "s3:DeleteObject", Resource: "\($b)/box-state/*"}]}')"
    if ! aws iam get-instance-profile --instance-profile-name rlvr-box >/dev/null 2>&1; then
      aws iam create-instance-profile --instance-profile-name rlvr-box >/dev/null
      aws iam add-role-to-instance-profile --instance-profile-name rlvr-box --role-name rlvr-box
    fi
    echo "roles rlvr-scheduler (policy run-rlvr-box) and rlvr-box (policy box-s3, instance profile rlvr-box) are set;" \
      "a new role takes some seconds to be usable"
    ;;
  lifecycle)
    if aws s3api get-bucket-lifecycle-configuration --bucket "$BUCKET" >"$tmp/old" 2>/dev/null \
      && [[ $(jq '[.Rules[] | select(.ID | startswith("rlvr-box-") | not)] | length' "$tmp/old") != 0 ]]; then
      echo "the bucket has lifecycle rules that are not from here; not replacing them:" >&2
      jq -c '.Rules[]' "$tmp/old" >&2
      exit 1
    fi
    aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration '{"Rules":[
      {"ID":"rlvr-box-state","Filter":{"Prefix":"box-state/"},"Status":"Enabled","Expiration":{"Days":7},"AbortIncompleteMultipartUpload":{"DaysAfterInitiation":2}},
      {"ID":"rlvr-box-files","Filter":{"Prefix":"box/"},"Status":"Enabled","Expiration":{"Days":90},"AbortIncompleteMultipartUpload":{"DaysAfterInitiation":2}}]}' >/dev/null
    aws s3api get-bucket-lifecycle-configuration --bucket "$BUCKET" --query 'Rules[].[ID,Filter.Prefix,Expiration.Days]' --output text
    ;;
  template)
    box "${1:-}"
    default_version "$FROM" "$tmp/from.json" || { echo "no launch template $FROM to copy the ssh keys and venv pins from (set FROM)" >&2; exit 1; }
    jq -r '.LaunchTemplateData.UserData' "$tmp/from.json" | base64 -d >"$tmp/from.userdata"
    mapfile -t keys < <(sed -n 's/^ *printf .%s\\n. "\([^"]*\)" "\([^"]*\)" >>.*authorized_keys$/\1\n\2/p' "$tmp/from.userdata")
    freeze=$(sed -n 's/^echo "\([A-Za-z0-9+\/=]*\)" | base64 -d | gunzip >.*vllm-venv-freeze\.txt$/\1/p' "$tmp/from.userdata")
    [[ ${#keys[@]} == 2 && ${keys[0]} == ssh-* && ${keys[1]} == ssh-* && -n $freeze ]] \
      || { echo "could not read two ssh keys and the venv pins from $FROM's user data" >&2; exit 1; }
    ami=${AMI:-$(jq -r '.LaunchTemplateData.ImageId' "$tmp/from.json")}
    eni=${ENI:-$(aws ec2 describe-network-interfaces --filters "Name=description,Values=rlvr box $zone" \
      --query 'NetworkInterfaces[0].NetworkInterfaceId' --output text)}
    [[ $eni == eni-* ]] || { echo "no network interface described \"rlvr box $zone\" (set ENI)" >&2; exit 1; }
    max='' no=0 volume=${VOLUME_GB:-2000}
    [[ -z $cr ]] && max=${MAX_MIN:-40} no=1 volume=${VOLUME_GB:-80}  # a test instance: no venv, ends itself
    sed -e "s|@END_EPOCH@|$(date -d "$end" +%s)|" -e "s|@AGENT_KEY@|${keys[0]}|" -e "s|@LAPTOP_KEY@|${keys[1]}|" \
      -e "s|@FREEZE_GZ_B64@|$freeze|" -e "s|@AUTO_S3@|s3://$BUCKET/box/$name|" -e "s|@MAX_MIN@|$max|" -e "s|@NO_VENV@|$no|" \
      scripts/aws/box_userdata.sh >"$tmp/userdata"
    ! grep -q '@[A-Z_0-9]*@' "$tmp/userdata" || { echo "a placeholder is left in the user data" >&2; exit 1; }
    bash -n "$tmp/userdata"
    (( $(stat -c %s "$tmp/userdata") < 16000 )) || { echo "the user data is over 16 kB" >&2; exit 1; }
    jq -n --arg ami "$ami" --arg itype "$itype" --arg eni "$eni" --arg name "$name" --arg cr "$cr" --argjson volume "$volume" \
      --arg userdata "$(base64 -w0 "$tmp/userdata")" '
      {ImageId: $ami, InstanceType: $itype, UserData: $userdata, IamInstanceProfile: {Name: "rlvr-box"},
       NetworkInterfaces: [{DeviceIndex: 0, NetworkInterfaceId: $eni}],
       TagSpecifications: [{ResourceType: "instance", Tags: [{Key: "Name", Value: $name}, {Key: "Project", Value: "rlvr-speedrun"}]}]}
      + if $cr == "" then
          {InstanceInitiatedShutdownBehavior: "terminate",
           BlockDeviceMappings: [{DeviceName: "/dev/sda1", Ebs: {VolumeSize: $volume, VolumeType: "gp3", DeleteOnTermination: true}}]}
        else
          {InstanceMarketOptions: {MarketType: "capacity-block"},
           CapacityReservationSpecification: {CapacityReservationTarget: {CapacityReservationId: $cr}},
           BlockDeviceMappings: [{DeviceName: "/dev/sda1", Ebs: {VolumeSize: $volume, VolumeType: "gp3", Iops: 6000, Throughput: 500, DeleteOnTermination: true}}]}
        end' >"$tmp/data.json"
    note="self-running: fetches s3://<bucket>/box/$name at boot (scripts/aws/box_auto.sh, commit $(git rev-parse --short HEAD))"
    if default_version "$name" "$tmp/old.json"; then
      v=$(aws ec2 create-launch-template-version --launch-template-name "$name" --version-description "$note" \
        --launch-template-data "file://$tmp/data.json" --query 'LaunchTemplateVersion.VersionNumber' --output text)
      aws ec2 modify-launch-template --launch-template-name "$name" --default-version "$v" >/dev/null
      echo "$name: version $v is now the default (it was $(jq -r .VersionNumber "$tmp/old.json"))"
    else
      aws ec2 create-launch-template --launch-template-name "$name" --version-description "$note" \
        --tag-specifications 'ResourceType=launch-template,Tags=[{Key=rlvr-box,Value=true},{Key=Project,Value=rlvr-speedrun}]' \
        --launch-template-data "file://$tmp/data.json" --query 'LaunchTemplate.LaunchTemplateId' --output text
      echo "$name: created"
    fi
    echo "$name: $itype in $zone, $ami, $eni, block end $end, user data $(stat -c %s "$tmp/userdata") bytes"
    ;;
  schedule)
    box "${1:-}"
    shift
    (( $# )) || set -- 1 5 10 25 60 180 360 540 720 900 1080
    default_version "$name" "$tmp/lt.json" || { echo "no launch template $name (run: box_auto.sh template)" >&2; exit 1; }
    target=$(jq -n --arg role "arn:aws:iam::$account:role/rlvr-scheduler" --arg name "$name" '
      {Arn: "arn:aws:scheduler:::aws-sdk:ec2:runInstances", RoleArn: $role,
       Input: ({LaunchTemplate: {LaunchTemplateName: $name}, MinCount: 1, MaxCount: 1} | tojson),
       RetryPolicy: {MaximumRetryAttempts: 3, MaximumEventAgeInSeconds: 600}}')
    for off in "$@"; do
      at=$(date -u -d "$start + $off min" +%FT%T)
      (( $(date -d "$at Z" +%s) < $(date -d "$end" +%s) - 3600 )) || { echo "skipped +$off min: under an hour before the block ends"; continue; }
      sname=$(printf '%s-m%04d' "$name" "$off")
      aws scheduler delete-schedule --name "$sname" >/dev/null 2>&1 || true
      aws scheduler create-schedule --name "$sname" --schedule-expression "at($at)" --schedule-expression-timezone UTC \
        --flexible-time-window Mode=OFF --action-after-completion NONE --target "$target" >/dev/null
      echo "$sname: RunInstances at $at UTC"
    done
    ;;
  unschedule)
    box "${1:-}"
    for sname in $(aws scheduler list-schedules --name-prefix "$name-m" --query 'Schedules[].Name' --output text); do
      aws scheduler delete-schedule --name "$sname" && echo "deleted $sname"
    done
    ;;
  dryrun)
    box "${1:-}"
    read -r key secret token < <(aws sts assume-role --role-arn "arn:aws:iam::$account:role/rlvr-scheduler" \
      --role-session-name box-auto-dryrun --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken]' --output text)
    out=$(AWS_ACCESS_KEY_ID=$key AWS_SECRET_ACCESS_KEY=$secret AWS_SESSION_TOKEN=$token aws ec2 run-instances --dry-run \
      --launch-template "LaunchTemplateName=$name" --count 1 2>&1) || true
    if grep -q DryRunOperation <<<"$out"; then
      echo "dry run ok: the rlvr-scheduler role may launch $name"
    else
      sed 's/Encoded authorization failure message: .*/(encoded message, decoded below)/' <<<"$out"
      msg=$(sed -n 's/.*Encoded authorization failure message: *//p' <<<"$out")
      [[ -n $msg ]] && aws sts decode-authorization-message --encoded-message "$msg" --query DecodedMessage --output text \
        | jq -c '{allowed, action: .context.action, resource: .context.resource}' | sed "s/$account/<account id>/g"
      echo "dry run FAILED"
      exit 1
    fi
    ;;
  show)
    box "${1:-}"
    [[ -n $cr ]] && aws ec2 describe-capacity-reservations --capacity-reservation-ids "$cr" --output text \
      --query 'CapacityReservations[0].[CapacityReservationId,InstanceType,AvailabilityZone,State,StartDate,EndDate]'
    if default_version "$name" "$tmp/lt.json"; then
      jq -r '.LaunchTemplateData.UserData' "$tmp/lt.json" | base64 -d >"$tmp/userdata"
      jq -r '"template \(.LaunchTemplateName) \(.LaunchTemplateId): default version \(.VersionNumber) (\(.VersionDescription // "no description")); "
        + "\(.LaunchTemplateData.InstanceType), \(.LaunchTemplateData.ImageId), \(.LaunchTemplateData.NetworkInterfaces[0].NetworkInterfaceId), "
        + "instance profile \(.LaunchTemplateData.IamInstanceProfile.Name // "NONE")"' "$tmp/lt.json"
      auto=$(sed -n 's/^auto="\(s3:[^"]*\)".*/\1/p' "$tmp/userdata" | sed "s/$account/<account id>/")
      echo "user data: self-running stage ${auto:-OFF (this version does not start by itself)}; block end $(date -u -d "@$(sed -n 's/^echo "\([0-9]*\)" >\/etc\/rlvr-box.end.*/\1/p' "$tmp/userdata")" +%FT%TZ)"
    else
      echo "template $name: NONE"
    fi
    echo "schedules:"
    for sname in $(aws scheduler list-schedules --name-prefix "$name-m" --query 'Schedules[].Name' --output text); do
      aws scheduler get-schedule --name "$sname" --query '[Name,ScheduleExpression,State,Target.Input]' --output text | sed 's/^/  /'
    done
    echo "instances (not terminated):"
    aws ec2 describe-instances --filters "Name=tag:Name,Values=$name" Name=instance-state-name,Values=pending,running,shutting-down,stopping,stopped \
      --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime,PublicIpAddress]' --output text | sed 's/^/  /'
    echo "RunInstances calls in the last ${HOURS:-6} h (CloudTrail, up to 15 min late):"
    aws cloudtrail lookup-events --lookup-attributes AttributeKey=EventName,AttributeValue=RunInstances \
      --start-time "$(date -u -d "-${HOURS:-6} hours" +%FT%TZ)" --query 'Events[].CloudTrailEvent' --output json \
      | jq -r '.[] | fromjson | "  \(.eventTime) \(.userIdentity.arn | split("/")[1:] | join("/")): \(.errorCode // "ok") \(.responseElements.instancesSet.items[0].instanceId // "") \((.errorMessage // "")[0:110])"' \
      | sed "s/$account/<account id>/g"
    ;;
  *)
    echo "unknown command $cmd" >&2
    exit 1
    ;;
esac
