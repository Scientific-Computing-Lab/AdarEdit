#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPL_DIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/adaredit_component_ablation_mpl_${USER:-user}}"

mkdir -p "$MPL_DIR"
export MPLCONFIGDIR="$MPL_DIR"

# Route A always verifies the canonical outputs shipped with the repository.
unset ADAREDIT_ABLATION_FULL ADAREDIT_ABLATION_RUNS

"$PYTHON_BIN" "$BASE/scripts/check_inputs.py"
"$PYTHON_BIN" "$BASE/scripts/summarize_ablations.py"
"$PYTHON_BIN" "$BASE/scripts/make_ablation_fig.py"
"$PYTHON_BIN" "$BASE/scripts/validate_outputs.py"

echo "[component ablation] PASS"
