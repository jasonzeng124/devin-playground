#!/bin/bash
# Start the unattended research agent (docs/research/AGENT.md) in tmux session "research" on the
# Engaging login node, unless it is already running. Permission checks are skipped at the
# maintainer's request (2026-10-22: auto mode kept stalling on confirmation prompts); the deny
# rules in .claude/settings.local.json still apply.
#
# Since 2026-10-29 (maintainer, to save usage): one wake per session, a fresh headless session at
# most every WAKE_H hours (default 6), effort high, no ultracode. State carries over only through
# the repo files (LOG, STATUS, PLAN, INBOX), as before.
#
# Model: Opus (maintainer 2026-10-25, after briefly trying Fable). Run this every 10 min as a
# keepalive; there is no crontab here, so the loop runs in its own tmux session:
#   tmux new-session -d -s research-watch 'while true; do bash ~/rlvr-screens/scripts/research/start_agent.sh; sleep 600; done'
set -u
A=$HOME/rlvr-screens/agent
mkdir -p "$A"
tmux has-session -t =research 2>/dev/null && exit 0  # "=": exact name (plain "research" also matches research-watch)
[[ -e $A/STOP ]] && exit 0  # touch agent/STOP to keep it from being restarted
WAKE_H=${WAKE_H:-6}
[[ -n $(find "$A/last_wake" -mmin -$(( WAKE_H * 60 )) 2>/dev/null) ]] && exit 0  # woke within WAKE_H hours
m=opus
touch "$A/last_wake"
echo "$(date -Is) starting research agent wake on $m" >>"$A/starts.log"
PROMPT='Autoresearch on RLVR v3 until 2026-11-19: this session is one wake. Read docs/research/AGENT.md in full first and follow its hard rules, do its "Each wake" steps once, then end. Keep quiet wakes short.'
tmux new-session -d -s research -c "$HOME/rlvr-screens" \
  "$HOME/.local/bin/claude -p --model "$m" --effort high --dangerously-skip-permissions -n research $(printf %q "$PROMPT") >>$A/wakes.log 2>&1"
