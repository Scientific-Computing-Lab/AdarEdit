#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPL_DIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/adaredit_training_dynamics_mpl_${USER:-user}}"

mkdir -p "$MPL_DIR"
export MPLCONFIGDIR="$MPL_DIR"
unset ADAREDIT_DYNAMICS_CHECKPOINTS

"$PYTHON_BIN" "$BASE/make_training_dynamics.py"
"$PYTHON_BIN" "$BASE/validate_outputs.py"

echo "[training dynamics] PASS"
