#!/bin/bash
# Start the unattended research agent (docs/research/AGENT.md) in tmux session "research" on the
# Engaging login node, unless it is already running. Safe to call from cron:
#   */30 * * * * $HOME/rlvr-screens/scripts/research/start_agent.sh
#   @reboot      $HOME/rlvr-screens/scripts/research/start_agent.sh
set -u
tmux has-session -t research 2>/dev/null && exit 0
[[ -e $HOME/rlvr-screens/agent/STOP ]] && exit 0  # touch agent/STOP to keep it from being restarted
mkdir -p "$HOME/rlvr-screens/agent"
echo "$(date -Is) starting research agent" >> "$HOME/rlvr-screens/agent/starts.log"
PROMPT='/loop Autoresearch on RLVR v3 until 2026-11-19. Read docs/research/AGENT.md in full first and follow its hard rules; each wake, do its "Each wake" steps. Keep quiet wakes short.'
tmux new-session -d -s research -c "$HOME/rlvr-screens" \
  "$HOME/.local/bin/claude --model opus --effort max --permission-mode auto -n research $(printf %q "$PROMPT"); sleep 600"
