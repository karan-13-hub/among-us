#!/usr/bin/env bash
# Worker that headlessly executes eval_qwen3_4b.ipynb via papermill.
#
# Designed to be called by launch_eval_bg.sh inside a detached tmux
# session, but also runnable directly. The script:
#   * activates the verl-py conda env
#   * loads the API-key env file used by the notebook
#   * runs papermill against a copy of the notebook so the source is
#     never modified
#   * streams stdout/stderr to a timestamped log under <EVAL_BASE>/runs/
#
# Usage: run_eval_bg.sh [NOTEBOOK_PATH]
#
# Environment (optional overrides):
#   CONDA_BASE   — conda installation root (default: $HOME/anaconda3)
#   CONDA_ENV    — env name (default: verl-py)
#   EVAL_BASE    — parent dir for runs/ output (default: $HOME/eval-among-us)
#   SECRETS_ENV  — if set and exists, sourced before papermill (API keys, etc.)
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NOTEBOOK="${1:-${_SCRIPT_DIR}/eval_qwen3_4b.ipynb}"
if [[ ! -f "$NOTEBOOK" ]]; then
  echo "ERROR: notebook not found: $NOTEBOOK" >&2
  exit 1
fi

CONDA_BASE="${CONDA_BASE:-/data/kmirakho/anaconda3}"
CONDA_ENV="${CONDA_ENV:-verl-py}"
EVAL_BASE="${EVAL_BASE:-/data/kmirakho/eval-cross-play-among-us}"
SECRETS_ENV="${SECRETS_ENV:-}"

NB_DIR="$(dirname "$NOTEBOOK")"
NB_STEM="$(basename "${NOTEBOOK%.ipynb}")"
TS="$(date +%Y%m%d_%H%M%S)"

RUN_DIR="${EVAL_BASE}/runs"
mkdir -p "$RUN_DIR"
OUT_NB="${RUN_DIR}/${NB_STEM}.${TS}.ipynb"
OUT_LOG="${RUN_DIR}/${NB_STEM}.${TS}.log"
PID_FILE="${RUN_DIR}/${NB_STEM}.current.pid"

echo "==> Background notebook run"
echo "  notebook : $NOTEBOOK"
echo "  out nb   : $OUT_NB"
echo "  out log  : $OUT_LOG"
echo "  conda env: $CONDA_ENV"
echo "  pid file : $PID_FILE"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [[ -n "$SECRETS_ENV" && -f "$SECRETS_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SECRETS_ENV"
  set +a
fi

export PYTHONUNBUFFERED=1
export AMONGUS_LOGPROBS_ENABLED="${AMONGUS_LOGPROBS_ENABLED:-1}"

echo "$$" > "$PID_FILE"

trap 'rm -f "$PID_FILE"' EXIT

cd "$NB_DIR"

# --log-output streams cell stdout/stderr to OUR stdout (which the
# launcher tees to OUT_LOG via |). --request-save-on-cell-execute writes
# the executed notebook after every cell so a kill leaves a partial
# checkpoint we can inspect. --kernel python3 matches the notebook's
# kernelspec (which points at the verl-py env we just activated).
papermill \
  --kernel python3 \
  --log-output \
  --progress-bar \
  --request-save-on-cell-execute \
  "$NOTEBOOK" \
  "$OUT_NB" \
  2>&1 | tee "$OUT_LOG"

EXIT_CODE="${PIPESTATUS[0]}"
echo ""
echo "==> papermill exited with code $EXIT_CODE"
echo "  executed notebook : $OUT_NB"
echo "  log               : $OUT_LOG"
exit "$EXIT_CODE"
