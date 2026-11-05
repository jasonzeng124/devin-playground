#!/usr/bin/env bash
# One-off (LOG 2026-11-05 01:45): copy r46's pretraining token file to the 11-05 box. Run as a CPU job.
set -euo pipefail
BOX=${BOX:?set BOX=zjason@<box ip>}
KNOWN_HOSTS=$HOME/rlvr-screens/agent/known_hosts_${BOX#*@}
export RSYNC_RSH="ssh -i $HOME/.ssh/rlvr_aws -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS -o BatchMode=yes -o ConnectTimeout=20"
$RSYNC_RSH "$BOX" mkdir -p orcd/pool/data/dm_math
rsync -a "$HOME/orcd/pool/data/dm_math/mix_0.2.bin" "$BOX:orcd/pool/data/dm_math/"
$RSYNC_RSH "$BOX" ls -l orcd/pool/data/dm_math/mix_0.2.bin
