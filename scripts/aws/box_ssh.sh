#!/usr/bin/env bash
# ssh to an AWS capacity-block box from Engaging with the agent's key, non-interactively. The box's
# host key goes to a known-hosts file of its own per box address (agent/known_hosts_<ip>), so a key
# left by an earlier instance on the same address (a TEST launch of box_launch.sh, or box 4, which
# reuses box 1's address) can't make BatchMode logins fail. scripts/aws/box_sync.sh uses the same
# options. For a new instance on a known address, delete that box's file first (or set KNOWN_HOSTS).
#
#   BOX=zjason@100.29.118.132 bash scripts/aws/box_ssh.sh 'cat /var/lib/rlvr-box.ready; nvidia-smi -L'
set -euo pipefail
BOX=${BOX:?set BOX=zjason@<box ip>}
KNOWN_HOSTS=${KNOWN_HOSTS:-$HOME/rlvr-screens/agent/known_hosts_${BOX#*@}}
exec ssh -i "$HOME/.ssh/rlvr_aws" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN_HOSTS" \
  -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 "$BOX" "$@"
