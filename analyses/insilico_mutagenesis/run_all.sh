#!/usr/bin/env bash
set -euo pipefail

ANALYSIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-auto}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/adaredit_matplotlib_${USER:-user}}"
export MPLCONFIGDIR

cd "$ANALYSIS_DIR"
mkdir -p "$MPLCONFIGDIR"

"$PYTHON_BIN" scripts/run_mutagenesis.py \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE"

"$PYTHON_BIN" scripts/make_figure.py

echo "PASS: analysis tables are in data/ and figures are in figures/"
