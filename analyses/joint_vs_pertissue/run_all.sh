#!/usr/bin/env bash
set -euo pipefail

ANALYSIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/adaredit_joint_comparison_mpl}"
export MPLCONFIGDIR
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON_BIN}" "${ANALYSIS_DIR}/scripts/build_comparison_data.py"
"${PYTHON_BIN}" "${ANALYSIS_DIR}/scripts/make_joint_comparison.py"

echo "PASS: joint-versus-per-tissue data and figures regenerated."
