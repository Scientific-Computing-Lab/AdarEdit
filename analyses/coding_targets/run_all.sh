#!/usr/bin/env bash
# Reproduce the coding-target inference, AUROC tables, and figures.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/adaredit_coding_matplotlib_${USER:-user}}"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

echo "[1/3] Generating predictions from the supplied trained checkpoints"
"$PYTHON" "$ROOT/scripts/generate_predictions.py" \
  --data-root "$ROOT/data" \
  --checkpoint-root "$ROOT/checkpoints" \
  --output-root "$ROOT/predictions" \
  --device "$DEVICE"

echo "[2/3] Building the per-gene AUROC table"
"$PYTHON" "$ROOT/scripts/build_summary.py" \
  --predictions-root "$ROOT/predictions" \
  --csv "$ROOT/results/auroc_summary.csv" \
  --json "$ROOT/results/auroc_summary.json"

echo "[3/3] Creating the AUROC figures"
"$PYTHON" "$ROOT/scripts/make_figures.py" \
  --summary "$ROOT/results/auroc_summary.csv" \
  --output-root "$ROOT/figures"

echo "PASS: coding-target predictions, AUROC table, and figures generated"
