#!/bin/bash
# User-data template for an AWS GPU box (one capacity block), filled in by scripts/aws/box_launch.sh.
# At first boot it sets up a user zjason with the same home layout as on Engaging (configs hold
# absolute /home/zjason/... paths), authorizes the Engaging agent's and the laptop's ssh keys, and
# builds ~/orcd/pool/vllm-venv from the Engaging venv's pinned package list. It needs no AWS
# credentials. The agent then copies the checkout, data and model cache over ssh and starts
# workers with scripts/aws/box_workers.sh. Progress: /var/log/rlvr-box.log; /var/lib/rlvr-box.ready
# appears when the venv works.
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
