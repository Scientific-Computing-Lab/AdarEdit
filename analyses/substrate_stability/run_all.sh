#!/usr/bin/env bash
set -euo pipefail

ANALYSIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$ANALYSIS_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPL_DIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/adaredit_stability_mpl_${USER:-user}}"

mkdir -p "$MPL_DIR"
export MPLCONFIGDIR="$MPL_DIR"

cd "$REPO_DIR"
"$PYTHON_BIN" analyses/substrate_stability/scripts/stability_performance.py
"$PYTHON_BIN" analyses/substrate_stability/scripts/make_stability_figure.py
"$PYTHON_BIN" analyses/substrate_stability/scripts/validate_outputs.py

echo "[Supplementary Fig. S6] PASS"
