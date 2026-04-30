#!/usr/bin/env bash
# Launch run_eval_bg.sh inside a detached tmux session so the run
# survives Cursor / SSH disconnects. Idempotent: if a session with the
# given name already exists, the script bails so we never accidentally
# spawn two parallel sweeps competing for the same vLLM servers.
#
# Usage:
#   ./launch_eval_bg.sh                       # default session name
#   ./launch_eval_bg.sh -s my-eval            # custom session name
#   ./launch_eval_bg.sh -n /path/to/x.ipynb   # custom notebook
#
# Monitor:
#   tmux ls
#   tmux attach -t among-us-eval     # detach: Ctrl-b d
#   tail -f ~/eval-among-us/runs/eval_qwen3_4b.*.log
#
# Stop:
#   ./kill_eval_bg.sh                # graceful (SIGTERM the tmux session)
set -euo pipefail

EVAL_BASE="${EVAL_BASE:-/data/kmirakho/eval-cross-play-among-us}"

SESSION="among-us-eval"
_LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
NOTEBOOK="${_LAUNCH_DIR}/eval_qwen3_4b.ipynb"

while getopts ":s:n:h" opt; do
  case "$opt" in
    s) SESSION="$OPTARG" ;;
    n) NOTEBOOK="$OPTARG" ;;
    h)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    \?) echo "Unknown option: -$OPTARG" >&2; exit 2 ;;
  esac
done

WORKER="$(cd "$(dirname "$0")" && pwd)/run_eval_bg.sh"
if [[ ! -x "$WORKER" ]]; then
  chmod +x "$WORKER"
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found on PATH; install tmux or run run_eval_bg.sh under nohup instead." >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists." >&2
  echo "  Attach: tmux attach -t $SESSION" >&2
  echo "  Or stop it first: ./kill_eval_bg.sh -s $SESSION" >&2
  exit 1
fi

echo "==> Starting tmux session '$SESSION'"
echo "    worker  : $WORKER"
echo "    notebook: $NOTEBOOK"

# -d: detached. -s: session name. The wrapper exec'd inside tmux runs
# the worker; once the worker exits we hold the pane open with `read`
# so the user can `tmux attach` post-hoc to inspect the final output.
tmux new-session -d -s "$SESSION" \
  "bash -lc '\"$WORKER\" \"$NOTEBOOK\"; ec=\$?; echo; echo \"[worker exited code \$ec; press Enter to close pane]\"; read -r _'"

# Brief wait so the worker has time to write its PID file / log header.
sleep 2

echo ""
echo "==> Launched. The run survives Cursor / shell exit."
echo "  Monitor (live attach):  tmux attach -t $SESSION    # detach: Ctrl-b d"
echo "  Tail log:               tail -f ${EVAL_BASE}/runs/*.log"
echo "  List sessions:          tmux ls"
echo "  Stop the run:           $(dirname "$0")/kill_eval_bg.sh -s $SESSION"

LATEST_LOG="$(ls -t "${EVAL_BASE}/runs"/*.log 2>/dev/null | head -1 || true)"
if [[ -n "$LATEST_LOG" ]]; then
  echo ""
  echo "  Latest log: $LATEST_LOG"
fi
