#!/usr/bin/env bash
# Runs ON a spot instance launched by the one-seed EC2 Fleet from scripts/aws/spot_launch.sh
# (systemd service). Trains one seed, pushes results to S3 every few minutes, then deletes its
# fleet and terminates the instance. If spot reclaims the instance, the fleet launches a
# replacement, which files the partial run under extra/ and starts the seed over. Needs
# nothing from the machine that launched it.
#
# /etc/rlvr.env (written by the launch user-data): BUCKET RUN SEED RECORD DEADLINE [EXTRA_FLAGS]
#
# V3=1 with SLOTS="<record>:<seed> ..." runs one seed per GPU on a multi-GPU instance
# (e.g. p5.48xlarge, 8x H100). Seed i of the list gets GPU i. SEED then just names the
# instance (box), and results go to runs/$RUN/<track dir>/seed_<n>/.
set -uo pipefail
source /etc/rlvr.env
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
cd /opt/rlvr
read -ra slots <<<"${SLOTS:-$RECORD:$SEED}"
slot_dest() {  # <record>:<seed> -> its S3 prefix
  if [[ -n "${SLOTS:-}" ]]; then echo "s3://$BUCKET/runs/$RUN/$(basename "$(dirname "${1%:*}")")/seed_${1##*:}"
  else echo "s3://$BUCKET/runs/$RUN/seed_${1##*:}"; fi
}
dest="s3://$BUCKET/runs/$RUN/$([[ -n "${SLOTS:-}" ]] && echo box || echo seed)_$SEED"  # logs and DONE
exec >>/var/log/rlvr-job.log 2>&1
status() { echo "$(date -u +%FT%TZ) $*" | tee -a /var/lib/rlvr/status.log; }
mkdir -p /var/lib/rlvr

token=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
iid=$(curl -s -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/instance-id)

push() {
  local sl rec s d
  for sl in "${slots[@]}"; do
    rec=${sl%:*} s=${sl##*:} d=$(slot_dest "$sl")
    [[ -d "$rec/seeds/$s" ]] && aws s3 sync --quiet "$rec/seeds/$s" "$d/seeds/$s" --exclude 'final/*'
    [[ -d "$rec/extra" ]] && aws s3 sync --quiet "$rec/extra" "$d/extra" --exclude '*' --include "interrupted_${s}_*" --include "oom_$s/*" --exclude '*/final/*'
  done
  aws s3 cp --quiet /var/lib/rlvr/status.log "$dest/status.log"
  aws s3 cp --quiet /var/log/rlvr-job.log "$dest/job.log"
  for f in /var/log/rlvr-slot*.log; do [[ -e $f ]] && aws s3 cp --quiet "$f" "$dest/$(basename "$f" | sed 's/^rlvr-//')"; done
}
finish() {
  trap - EXIT
  status "finish: $*"
  push
  echo "$*" | aws s3 cp --quiet - "$dest/DONE"  # a replacement instance stops at once
  aws ec2 delete-launch-template --launch-template-name "rlvr-$RUN-s$SEED" >/dev/null 2>&1
  # delete the fleet before terminating, or it would replace this instance
  fleet=$(aws ec2 describe-instances --instance-ids "$iid" --query 'Reservations[0].Instances[0].Tags[?Key==`aws:ec2:fleet-id`]|[0].Value' --output text)
  [[ "$fleet" == fleet-* ]] && aws ec2 delete-fleets --fleet-ids "$fleet" --terminate-instances >/dev/null
  aws ec2 terminate-instances --instance-ids "$iid" >/dev/null || shutdown -h now
  exit 0
}

# any unexpected exit (set -u, a failed command in the setup) must still end the instance
trap 'finish "job script exited unexpectedly"' EXIT

# backstop: however the job ends up, the instance is gone by DEADLINE (set at first boot)
( while sleep 60; do (( $(date +%s) > DEADLINE )) && finish "backstop deadline reached"; done ) &
( while sleep 300; do push; done ) &

# a replacement launched after this seed already finished (e.g. delete-fleets failed)
aws s3 ls "$dest/DONE" >/dev/null 2>&1 && finish "already done"

# earlier instances of this seed that spot reclaimed left partial runs in S3: keep them
# under extra/ (RULES: no dropped runs) and start over
interrupted=0
for sl in "${slots[@]}"; do
  s=${sl##*:} d=$(slot_dest "$sl")
  n=$(aws s3 ls "$d/extra/" 2>/dev/null | grep -c "interrupted_" || true)
  if aws s3 ls "$d/seeds/$s/train_log.jsonl" >/dev/null 2>&1 || aws s3 ls "$d/seeds/$s/pretrain_log.jsonl" >/dev/null 2>&1; then
    n=$((n + 1))
    aws s3 mv --quiet --recursive "$d/seeds/$s" "$d/extra/interrupted_${s}_$n"
  fi
  (( n > interrupted )) && interrupted=$n
done
status "start on $iid after $interrupted interruptions: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
(( interrupted >= 3 )) && finish "giving up after 3 interruptions"

if [[ "${V3:-0}" == 1 ]]; then
  # benchmark v3: the reference software (benchmark.V3_REFERENCE_SOFTWARE) in a fresh uv env on the base DLAMI
  export HOME=/root PATH=/root/.local/bin:$PATH
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || finish "uv install failed"
  uv venv -q -p 3.11 /opt/v3env && source /opt/v3env/bin/activate || finish "venv failed"
  uv pip install -q "vllm==0.30.0" "transformers==5.17.0" pyarrow numpy accelerate huggingface_hub && uv pip install -q --no-deps -e . \
    || finish "pip install failed"
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
  export PYTHON=python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CKPT_DIR=/opt/ckpts CONFIRM_LIMIT=${CONFIRM_LIMIT:-}
  status "env: $(python -c 'import torch, transformers, vllm; print("torch", torch.__version__, "transformers", transformers.__version__, "vllm", vllm.__version__)') driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
  tracks=$(for sl in "${slots[@]}"; do python -c "import json; print(json.load(open('${sl%:*}/config.json'))['v3']['track'])"; done | sort -u)
  if grep -q D <<<"$tracks"; then
    python scripts/prepare_track_d_data.py >>/var/log/rlvr-job.log 2>&1 || finish "data prep failed"
    status "track D data ready"
  fi
  if grep -q A <<<"$tracks"; then
    python -c "from huggingface_hub import snapshot_download; from rlvr_speedrun import benchmark as b; snapshot_download(b.V3_TRACK_A_MODEL, revision=b.V3_TRACK_A_REVISION)" \
      || finish "model download failed"
  fi
  run_slot() {
    if CUDA_VISIBLE_DEVICES=$1 bash scripts/run_seed_v3.sh "${2%:*}" "${2##*:}" >>"/var/log/rlvr-slot$1.log" 2>&1
    then status "ok $2"; else status "FAILED $2"; fi
  }
  pids=()
  for i in "${!slots[@]}"; do run_slot "$i" "${slots[$i]}" & pids+=($!); done
  wait "${pids[@]}"
  finish "done"
fi

# the AMI's PyTorch env: uv's standalone Python had no working triton on this AMI (2026-09)
set +u; source /opt/pytorch/bin/activate; set -u  # the AMI activate script reads unset variables
pip install -q "transformers==4.57.6" numpy accelerate pytest && pip install -q --no-deps -e . || finish "pip install failed"
export PYTHON=python DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
status "env: $(python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)')"

# shellcheck disable=SC2086
if bash scripts/run_seed.sh "$RECORD" "$SEED" ${EXTRA_FLAGS:-}; then
  status "ok seed $SEED"
elif grep -q "OutOfMemoryError\|CUDA out of memory" "$RECORD/seeds/$SEED/stdout.log"; then
  status "OOM seed $SEED -> retry with --micro-batch-size 32 --eval-batch-size 250"
  mkdir -p "$RECORD/extra/oom_$SEED" && mv "$RECORD/seeds/$SEED"/* "$RECORD/extra/oom_$SEED/"
  # shellcheck disable=SC2086
  bash scripts/run_seed.sh "$RECORD" "$SEED" ${EXTRA_FLAGS:-} --micro-batch-size 32 --eval-batch-size 250 \
    && status "ok seed $SEED (small batches)" || status "FAILED seed $SEED (small batches)"
else
  status "FAILED seed $SEED"
fi
finish "done"
