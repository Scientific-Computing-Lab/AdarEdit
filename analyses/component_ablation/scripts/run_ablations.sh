#!/usr/bin/env bash
# Retrain the full Liver bio-aware model and all six ablations.
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$BUNDLE/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAINER="$BUNDLE/pipeline/train_strict_long.py"
DATA_ROOT="$REPO/data/human"
CACHE_ROOT="${CACHE_ROOT:-$BUNDLE/cache}"
OUT_ROOT="${OUT_ROOT:-$BUNDLE/recomputed_runs}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_THREADS="${NUM_THREADS:-8}"

mkdir -p "$CACHE_ROOT" "$OUT_ROOT"

run_ablation() {
    local tag="$1"
    echo "=== Liver ablation: $tag ==="
    "$PYTHON_BIN" "$TRAINER" \
        --variant bioaware \
        --context Liver \
        --data-root "$DATA_ROOT" \
        --cache-root "$CACHE_ROOT" \
        --out-root "$OUT_ROOT" \
        --bio-ablation "$tag" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --num-threads "$NUM_THREADS" \
        --checkpoint-every 100 \
        --seed 42 \
        --resume
}

run_ablation full
run_ablation no_neighbors
run_ablation no_geometry
run_ablation no_seq_branch
run_ablation untyped_edges
run_ablation no_pair_edges
run_ablation shuffle_pair_edges

export ADAREDIT_ABLATION_FULL="$OUT_ROOT/bioaware_Liver/summary.json"
export ADAREDIT_ABLATION_RUNS="$OUT_ROOT/ablations/Liver"
"$PYTHON_BIN" "$BUNDLE/scripts/summarize_ablations.py"
"$PYTHON_BIN" "$BUNDLE/scripts/make_ablation_fig.py"
"$PYTHON_BIN" "$BUNDLE/scripts/validate_outputs.py"

echo "[component ablation] PASS"
