#!/bin/bash
# Start the unattended research agent (docs/research/AGENT.md) in tmux session "research" on the
# Engaging login node, unless it is already running. Permission checks are skipped at the
# maintainer's request (2026-10-22: auto mode kept stalling on confirmation prompts); the deny
# rules in .claude/settings.local.json still apply.
#
# The agent wakes hourly (the longest timer) but does a full wake only every 6 hours unless
# something needs it (maintainer 2026-10-29, to save usage; no ultracode). Every wake re-reads the
# whole context, so auto-compact is capped at COMPACT_WINDOW tokens (default 150k) to keep wakes
# cheap (idle compaction is behind a server-side flag, so it can't be relied on).
# If agent/session_id holds a session id, the agent resumes that conversation, so a restart keeps
# its context.
#
# Model: Fable 5.2 at effort xhigh since 2026-11-10 (maintainer). Before that: Opus 5.5 from
# 2026-10-25, at effort high from 2026-10-29. The run ends 2026-12-19 (extended from 2026-11-19
# on 2026-11-10).
#
# Run this every 10 min as a keepalive (crontab on login009 since 2026-11-02). It does two things:
#   - no tmux session: start the agent;
#   - session up but idle: re-issue the /loop command. The harness ends a self-paced /loop after
#     7 days (it did on 2026-11-09) and the session stays up, so the first check misses that.
#     The agent wakes at least hourly and every wake writes to its transcript, so a transcript
#     untouched for STALE_MIN minutes (default 120) means the loop has ended.
# DRY_RUN=1 prints what the idle check would do without sending anything.
set -u
A=$HOME/rlvr-screens/agent
END=2026-12-19
mkdir -p "$A"
[[ -e $A/STOP ]] && exit 0  # touch agent/STOP to keep it from being restarted or re-issued
PROMPT='/loop Autoresearch on RLVR v3 until 2026-12-19. Read docs/research/AGENT.md in full first and follow its hard rules; each wake, do its "Each wake" steps. Keep quiet wakes short.'
# "=": exact name (plain "research" also matches research-watch)
if tmux has-session -t =research 2>/dev/null; then
  [[ $(date +%F) > $END ]] && exit 0
  s=${STALE_MIN:-120}
  t=$(ls -t "$HOME/.claude/projects/${HOME//\//-}-rlvr-screens"/*.jsonl 2>/dev/null | head -1)
  [[ -n $t && -n $(find "$t" -mmin +"$s") ]] || exit 0
  # at most one re-issue per STALE_MIN
  [[ -e $A/last_nudge && -z $(find "$A/last_nudge" -mmin +"$s") ]] && exit 0
  if [[ -n ${DRY_RUN:-} ]]; then echo "would re-issue /loop (transcript $t idle over $s min)"; exit 0; fi
  touch "$A/last_nudge"
  echo "$(date -Is) session idle for over $s min; re-issuing /loop" >>"$A/starts.log"
  tmux send-keys -t =research: C-u
  tmux send-keys -t =research: -l "$PROMPT"
  sleep 1
  tmux send-keys -t =research: Enter
  exit 0
fi
m=claude-fable-5-2
e=xhigh
echo "$(date -Is) starting research agent on $m (effort $e)" >>"$A/starts.log"
resume=""
[[ -s $A/session_id ]] && resume="--resume $(<"$A/session_id")"
tmux new-session -d -s research -c "$HOME/rlvr-screens" \
  "CLAUDE_CODE_AUTO_COMPACT_WINDOW=${COMPACT_WINDOW:-150000} $HOME/.local/bin/claude $resume --model "$m" --effort "$e" --dangerously-skip-permissions -n research $(printf %q "$PROMPT"); sleep 600"
