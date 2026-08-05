#!/usr/bin/env bash
set -euo pipefail

ANALYSIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${ANALYSIS_DIR}/scripts/check_joint_leakage.py"
"${PYTHON_BIN}" "${ANALYSIS_DIR}/scripts/validate_outputs.py"
"${PYTHON_BIN}" "${ANALYSIS_DIR}/scripts/make_combined_control.py"
"${PYTHON_BIN}" "${ANALYSIS_DIR}/scripts/make_combined_control_figure.py"

echo "PASS: joint-model data, supplied outputs and Combined control validated."
