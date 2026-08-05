#!/usr/bin/env bash
set -euo pipefail

ANALYSIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$ANALYSIS_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${ADAREDIT_THRESHOLD_DEVICE:-auto}"
WORK_DIR="${ADAREDIT_THRESHOLD_WORK_DIR:-${TMPDIR:-/tmp}/adaredit_threshold_relaxation}"

cd "$ANALYSIS_DIR"
mkdir -p "$WORK_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WORK_DIR/matplotlib}"

BUILD_ARGS=()
if [[ -n "${ADAREDIT_EDITING_LEVEL_DIR:-}" ]]; then
  BUILD_ARGS+=(--raw-dir "$ADAREDIT_EDITING_LEVEL_DIR")
fi

INFERENCE_ARGS=()
if [[ "${ADAREDIT_THRESHOLD_RESUME:-0}" == "1" ]]; then
  INFERENCE_ARGS+=(--resume)
fi

"$PYTHON_BIN" scripts/build_full_cohorts.py "${BUILD_ARGS[@]}"
"$PYTHON_BIN" scripts/run_analysis.py \
  --device "$DEVICE" \
  --work-dir "$WORK_DIR" \
  "${INFERENCE_ARGS[@]}"
"$PYTHON_BIN" scripts/make_threshold_figures.py
cp figures/threshold_relaxation.png "$REPO_DIR/manuscript/threshold_relaxation.png"
"$PYTHON_BIN" scripts/validate_outputs.py

echo "PASS: threshold-relaxation analysis finished."
