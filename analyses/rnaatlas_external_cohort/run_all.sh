#!/usr/bin/env bash
set -euo pipefail

ANALYSIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPL_DIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/adaredit_rnaatlas_mpl_${USER:-user}}"

mkdir -p "$MPL_DIR"
export MPLCONFIGDIR="$MPL_DIR"

cd "$ANALYSIS_DIR"
"$PYTHON_BIN" scripts/make_concordance_fig.py

echo "PASS: RNAAtlas label-concordance figure regenerated."
