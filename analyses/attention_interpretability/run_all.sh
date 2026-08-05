#!/usr/bin/env bash
# Reproduce the complete attention-interpretability analysis.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/adaredit_attention_v2t_matplotlib_${USER:-user}}"
export MPLCONFIGDIR
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$MPLCONFIGDIR"

echo "[1/9] Verifying the dedicated software environment"
"$PYTHON" "$ROOT/scripts/check_environment.py"

echo "[2/9] Extracting validation attention"
"$PYTHON" "$ROOT/scripts/extract_attention.py" \
  --split valid --device "$DEVICE"

echo "[3/9] Extracting test attention and verifying shipped predictions"
"$PYTHON" "$ROOT/scripts/extract_attention.py" \
  --split test --device "$DEVICE"

echo "[4/9] Fitting XGBoost on validation and evaluating test"
"$PYTHON" "$ROOT/scripts/train_attention_probe.py"

echo "[5/9] Rendering panels B-G"
"$PYTHON" "$ROOT/scripts/make_figure.py"

echo "[6/9] Running the positional-availability sensitivity analysis"
"$PYTHON" "$ROOT/scripts/run_availability_control.py"

echo "[7/9] Rendering the positional-availability supplementary figure"
"$PYTHON" "$ROOT/scripts/make_availability_figure.py"

echo "[8/9] Validating positional-availability outputs"
"$PYTHON" "$ROOT/scripts/validate_availability_control.py"

echo "[9/9] Validating the complete analysis against the reference results"
"$PYTHON" "$ROOT/scripts/validate_outputs.py"

echo "PASS: attention-interpretability analysis completed"
