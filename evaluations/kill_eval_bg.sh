#!/usr/bin/env bash
# Stop a background eval tmux session cleanly.
#
# 1. Sends SIGINT to processes inside the tmux pane (lets papermill /
#    the kernel checkpoint the partial executed notebook on the way out).
# 2. Kills the tmux session itself.
# 3. Optionally also kills lingering vLLM servers with -V.
#
# Usage:
#   ./kill_eval_bg.sh                   # default session 'among-us-eval'
#   ./kill_eval_bg.sh -s my-eval        # custom session
#   ./kill_eval_bg.sh -V                # also kill vLLM servers
set -euo pipefail

SESSION="among-us-eval"
KILL_VLLM=0

while getopts ":s:Vh" opt; do
  case "$opt" in
    s) SESSION="$OPTARG" ;;
    V) KILL_VLLM=1 ;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; exit 2 ;;
  esac
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "==> Sending SIGINT to processes in tmux session '$SESSION'..."
  PANE_PID="$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' 2>/dev/null | head -1 || true)"
  if [[ -n "$PANE_PID" ]]; then
    pkill -INT -P "$PANE_PID" 2>/dev/null || true
    sleep 3
    pkill -TERM -P "$PANE_PID" 2>/dev/null || true
    sleep 1
  fi
  echo "==> Killing tmux session '$SESSION'..."
  tmux kill-session -t "$SESSION" 2>/dev/null || true
else
  echo "tmux session '$SESSION' not found (already stopped)."
fi

if [[ "$KILL_VLLM" -eq 1 ]]; then
  echo "==> Killing vLLM api_server processes..."
  pkill -f "vllm.entrypoints.openai.api_server" || true
  sleep 1
  pgrep -af "vllm.entrypoints.openai.api_server" || echo "  no vLLM processes remain"
fi

echo "Done."
