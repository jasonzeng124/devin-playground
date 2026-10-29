#!/bin/bash
# Start the unattended research agent (docs/research/AGENT.md) in tmux session "research" on the
# Engaging login node, unless it is already running. Permission checks are skipped at the
# maintainer's request (2026-10-22: auto mode kept stalling on confirmation prompts); the deny
# rules in .claude/settings.local.json still apply.
#
# Since 2026-10-29 (maintainer, to save usage): effort high, no ultracode. The agent wakes hourly
# (the longest timer) but does a full wake only every 6 hours unless something needs it. Every wake
# re-reads the whole context, so auto-compact is capped at COMPACT_WINDOW tokens (default 150k)
# to keep wakes cheap (idle compaction is behind a server-side flag, so it can't be relied on).
# If agent/session_id holds a session id, the agent resumes that conversation, so a restart keeps
# its context.
#
# Model: Opus (maintainer 2026-10-25, after briefly trying Fable). Run this every 10 min as a
# keepalive; there is no crontab here, so the loop runs in its own tmux session:
#   tmux new-session -d -s research-watch 'while true; do bash ~/rlvr-screens/scripts/research/start_agent.sh; sleep 600; done'
set -u
A=$HOME/rlvr-screens/agent
mkdir -p "$A"
tmux has-session -t =research 2>/dev/null && exit 0  # "=": exact name (plain "research" also matches research-watch)
[[ -e $A/STOP ]] && exit 0  # touch agent/STOP to keep it from being restarted
m=opus
echo "$(date -Is) starting research agent on $m" >>"$A/starts.log"
resume=""
[[ -s $A/session_id ]] && resume="--resume $(<"$A/session_id")"
PROMPT='/loop Autoresearch on RLVR v3 until 2026-11-19. Read docs/research/AGENT.md in full first and follow its hard rules; each wake, do its "Each wake" steps. Keep quiet wakes short.'
tmux new-session -d -s research -c "$HOME/rlvr-screens" \
  "CLAUDE_CODE_AUTO_COMPACT_WINDOW=${COMPACT_WINDOW:-150000} $HOME/.local/bin/claude $resume --model "$m" --effort high --dangerously-skip-permissions -n research $(printf %q "$PROMPT"); sleep 600"
