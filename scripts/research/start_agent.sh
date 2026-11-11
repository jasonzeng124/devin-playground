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
# Model: Opus 5.5 at effort high (from 2026-10-25, at effort high from 2026-10-29), except for
# 2026-11-10, when the maintainer tried Fable 5.2 at effort xhigh from about 10:00 and put it back
# on Opus that night. The run ends 2026-12-19 (extended from 2026-11-19 on 2026-11-10).
#
# Run this every 10 min as a keepalive (crontab on login009 since 2026-11-02). It does two things:
#   - no tmux session: start the agent;
#   - session up but idle: re-issue the /loop command. The harness ends a self-paced /loop after
#     7 days (it did on 2026-11-09) and the session stays up, so the first check misses that.
#     The agent wakes at least hourly and every wake adds entries to its transcript, so a
#     transcript whose last entry is older than STALE_MIN minutes (default 120) means the loop
#     has ended. The file's mtime can't be used for this: an idle session still touches its
#     transcript about once an hour (found 2026-11-11, when 12 idle hours went unnoticed).
# Low-usage window: while agent/slow_until holds a time that is still ahead (anything `date -d`
# reads, e.g. "2026-11-11 18:00"), the agent ends its loop after every wake (INBOX tells it to) and
# the idle check waits SLOW_MIN minutes (default 300) instead of STALE_MIN. So the agent wakes
# about every 5 hours. Delete the file, or let the time pass, to go back to normal.
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
  if [[ -s $A/slow_until ]] && u=$(date -d "$(<"$A/slow_until")" +%s 2>/dev/null) && (($(date +%s) < u)); then
    s=${SLOW_MIN:-300}
  fi
  t=$(ls -t "$HOME/.claude/projects/${HOME//\//-}-rlvr-screens"/*.jsonl 2>/dev/null | head -1)
  [[ -n $t ]] || exit 0
  # time of the last timestamped entry (UTC, ISO); no entry found: do nothing
  last=$(tail -c 8000000 "$t" | grep -o '"timestamp":"20[^"]*"' | tail -1 | cut -d'"' -f4)
  [[ -n $last ]] && lt=$(date -d "$last" +%s 2>/dev/null) || exit 0
  (($(date +%s) - lt > s * 60)) || exit 0
  # at most one re-issue per STALE_MIN (or SLOW_MIN)
  [[ -e $A/last_nudge && -z $(find "$A/last_nudge" -mmin +"$s") ]] && exit 0
  if [[ -n ${DRY_RUN:-} ]]; then echo "would re-issue /loop (last transcript entry $last, over $s min ago)"; exit 0; fi
  touch "$A/last_nudge"
  echo "$(date -Is) session idle for over $s min; re-issuing /loop" >>"$A/starts.log"
  tmux send-keys -t =research: C-u
  tmux send-keys -t =research: -l "$PROMPT"
  sleep 1
  tmux send-keys -t =research: Enter
  exit 0
fi
m=claude-opus-5-5
e=high
echo "$(date -Is) starting research agent on $m (effort $e)" >>"$A/starts.log"
resume=""
[[ -s $A/session_id ]] && resume="--resume $(<"$A/session_id")"
tmux new-session -d -s research -c "$HOME/rlvr-screens" \
  "CLAUDE_CODE_AUTO_COMPACT_WINDOW=${COMPACT_WINDOW:-150000} $HOME/.local/bin/claude $resume --model "$m" --effort "$e" --dangerously-skip-permissions -n research $(printf %q "$PROMPT"); sleep 600"
