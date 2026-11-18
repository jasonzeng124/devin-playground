#!/bin/bash
# User-data template for an AWS GPU box (one capacity block), filled in by scripts/aws/box_launch.sh.
# At first boot it sets up a user zjason with the same home layout as on Engaging (configs hold
# absolute /home/zjason/... paths), authorizes the Engaging agent's and the laptop's ssh keys, and
# builds ~/orcd/pool/vllm-venv from the Engaging venv's pinned package list. As filled by box_launch.sh it
# needs no AWS credentials: the agent then copies the checkout, data and model cache over ssh and starts
# workers with scripts/aws/box_workers.sh. As filled by scripts/aws/box_auto.sh the box runs by itself (see
# the stage below). Progress: /var/log/rlvr-box.log; /var/lib/rlvr-box.ready appears when the venv works.
exec >>/var/log/rlvr-box.log 2>&1
set -x
echo "@END_EPOCH@" >/etc/rlvr-box.end  # the block ends here; AWS may start terminating 30 min earlier
id zjason || useradd -m -s /bin/bash zjason
echo 'zjason ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/zjason
for u in zjason ubuntu; do
  install -d -m 700 -o $u -g $u /home/$u/.ssh
  printf '%s\n' "@AGENT_KEY@" "@LAPTOP_KEY@" >>/home/$u/.ssh/authorized_keys
  chown $u:$u /home/$u/.ssh/authorized_keys && chmod 600 /home/$u/.ssh/authorized_keys
done
install -d -o zjason -g zjason /home/zjason/orcd /home/zjason/orcd/pool /home/zjason/orcd/pool/data \
  /home/zjason/orcd/pool/rlvr-ckpts /home/zjason/.cache /home/zjason/.cache/huggingface /home/zjason/rlvr-screens
echo "@FREEZE_GZ_B64@" | base64 -d | gunzip >/home/zjason/vllm-venv-freeze.txt
chown zjason:zjason /home/zjason/vllm-venv-freeze.txt
nvidia-smi
# Self-running box: scripts/aws/box_auto.sh fills these three (scripts/aws/box_launch.sh leaves them as they are,
# which keeps this stage off). It fetches the staged code from S3 with the instance profile's credentials and starts
# scripts/aws/box_boot.sh, which fetches the rest, starts the workers and sends the results to S3.
auto="@AUTO_S3@" max_min="@MAX_MIN@" no_venv="@NO_VENV@"
[[ $max_min =~ ^[0-9]+$ ]] && shutdown -h +"$max_min"  # a test instance ends itself whatever happens
if [[ $auto == s3://* ]]; then
  token=$(curl -s -m 5 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
  region=$(curl -s -m 5 -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/placement/region)
  sudo -iu zjason bash -s "$auto" "${region:-us-east-1}" <<'EOF'
set -x
export AWS_DEFAULT_REGION=$2
for try in 1 2 3 4 5 6 7 8 9; do  # the instance profile's credentials can take a moment to appear
  aws s3 cp --only-show-errors "$1/in/code.tgz" ~/code.tgz && break
  sleep 20
done
tar xzf ~/code.tgz -C ~/rlvr-screens && cd ~/rlvr-screens \
  && setsid nohup bash scripts/aws/box_boot.sh "$1" >>~/box_boot.log 2>&1 </dev/null &
EOF
fi
[[ $no_venv == 1 ]] && exit 0  # a test instance without a GPU: no venv
sudo -iu zjason bash <<'EOF'
set -x
curl -LsSf https://astral.sh/uv/install.sh | sh
uv=~/.local/bin/uv py=~/orcd/pool/vllm-venv/bin/python
$uv venv -q -p 3.11 ~/orcd/pool/vllm-venv
$uv pip install -q --python $py -r ~/vllm-venv-freeze.txt \
  || $uv pip install -q --python $py "vllm==0.30.0" "transformers==5.17.0" pyarrow numpy accelerate huggingface_hub
$py -c 'import torch, vllm, transformers; print("torch", torch.__version__, torch.cuda.get_device_name(0), "vllm", vllm.__version__, "transformers", transformers.__version__)' \
  && touch ~/venv.ok
EOF
[[ -e /home/zjason/venv.ok ]] && date -u +%FT%TZ >/var/lib/rlvr-box.ready
