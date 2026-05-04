#!/usr/bin/env bash
# Run eval_cross_gemma_26b_vs_deepseek_qwen32b.ipynb via papermill.
#
# Usage:
#   ./run_cross_gemma_26b_vs_deepseek_qwen32b.sh              # foreground
#   nohup ./run_cross_gemma_26b_vs_deepseek_qwen32b.sh &      # background
#
# Override env vars to customise paths:
#   CONDA_BASE   conda installation root
#   CONDA_ENV    conda env name (default: verl-gemma)
#   EVAL_BASE    output root (default: /home/yjangir1/scratchhbharad2/users/yjangir1/karan/eval-cross-play-among-us)
#   SECRETS_ENV  optional env-file with OPENAI_API_KEY etc.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NOTEBOOK="${_SCRIPT_DIR}/eval_cross_gemma_26b_vs_deepseek_qwen32b.ipynb"

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "ERROR: notebook not found: $NOTEBOOK" >&2
  exit 1
fi

CONDA_BASE="${CONDA_BASE:-/home/yjangir1/scratchhbharad2/users/yjangir1/miniconda3}"
CONDA_ENV="${CONDA_ENV:-verl-gemma}"
EVAL_BASE="${EVAL_BASE:-/home/yjangir1/scratchhbharad2/users/yjangir1/karan/eval-cross-play-among-us}"
SECRETS_ENV="${SECRETS_ENV:-}"

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
echo "  conda env: $CONDA_ENV @ $CONDA_BASE"

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
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export VLLM_LOGGING_LEVEL=DEBUG

echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

# cd to the evaluations dir so relative imports (utils, evaluations.*) resolve
cd "$_SCRIPT_DIR"

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
