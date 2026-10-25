#!/bin/bash
# Start the unattended research agent (docs/research/AGENT.md) in tmux session "research" on the
# Engaging login node, unless it is already running. Permission checks are skipped at the
# maintainer's request (2026-10-22: auto mode kept stalling on confirmation prompts); the deny
# rules in .claude/settings.local.json still apply. Ultracode (workflow orchestration) is on at the
# maintainer's request (2026-10-25, with the Max 20x plan).
#
# Model: Fable, but Opus while Fable's usage limit is used up (maintainer 2026-10-25: "switch to
# opus if fable limit is used"). When the agent's screen shows a usage-limit message under Fable,
# this script restarts it on Opus for 12 h, then switches back to Fable the next time the agent
# is idle between wakes. Run it every 10 min; there is no crontab here, so the watch loop runs in
# its own tmux session:
#   tmux new-session -d -s research-watch 'while true; do bash ~/rlvr-screens/scripts/research/start_agent.sh; sleep 600; done'
set -u
A=$HOME/rlvr-screens/agent
mkdir -p "$A"
model() { cat "$A/model" 2>/dev/null; }
if tmux has-session -t research 2>/dev/null; then
  pane=$(tmux capture-pane -pt research | grep -v '^\s*$' | tail -12)
  if [[ $(model) == fable ]] && grep -qiE 'usage limit|limit reached|hit your( [a-z0-9. ]+)? limit|limit (will )?resets?' <<<"$pane"; then
    echo "$(date -Is) Fable usage limit on screen; restarting on Opus for 12 h" >>"$A/starts.log"
    date -d '+12 hours' +%s >"$A/fable_off_until"
  elif [[ $(model) == opus ]] && (( $(date +%s) >= $(cat "$A/fable_off_until" 2>/dev/null || echo 0) )) \
      && grep -qE '· done [0-9]' <<<"$pane"; then
    echo "$(date -Is) Fable pause over and agent idle; restarting on Fable" >>"$A/starts.log"
  else
    exit 0
  fi
  tmux kill-session -t research
fi
[[ -e $A/STOP ]] && exit 0  # touch agent/STOP to keep it from being restarted
m=fable
(( $(date +%s) < $(cat "$A/fable_off_until" 2>/dev/null || echo 0) )) && m=opus
echo "$m" >"$A/model"
echo "$(date -Is) starting research agent on $m" >>"$A/starts.log"
PROMPT='/loop Autoresearch on RLVR v3 until 2026-11-19. Read docs/research/AGENT.md in full first and follow its hard rules; each wake, do its "Each wake" steps. Keep quiet wakes short.'
tmux new-session -d -s research -c "$HOME/rlvr-screens" \
  "$HOME/.local/bin/claude --model $m --effort max --dangerously-skip-permissions --settings '{\"ultracode\": true}' -n research $(printf %q "$PROMPT"); sleep 600"
