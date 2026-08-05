#!/usr/bin/env bash
# Reproduce the target-centered triplet-feature baseline (logistic regression /
# linear SVM) on the human Liver split. One command, ~10 s (linear model, no GPU).
set -euo pipefail
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPTS/../../.." && pwd)"
RESULTS="$REPO/analyses/triplet_baseline/results"
WORK="$REPO/analyses/triplet_baseline/work"
TISSUE="${1:-Liver}"

# 1) build metadata CSVs (full_seq, structure, L, yes_no) from the human jsonl.
#    These go to work/, never into data/human/ -- the canonical metadata there uses
#    a different schema (pair_prefix, yes_no, split) and records the split provenance.
python3 "$SCRIPTS/build_metadata.py" "$TISSUE" --out_dir "$WORK"

# 2) run the triplet baseline: the two linear heads plus the RBF-kernel SVM
#    (the kernel used by Xue et al. 2005, the method this control is modelled on)
mkdir -p "$RESULTS"
for MODEL in logreg linear_svm rbf_svm; do
  python3 "$SCRIPTS/train_triplet_baseline.py" \
    --split_root "$WORK" \
    --tissue "$TISSUE" --model "$MODEL" \
    --out              "$RESULTS/${TISSUE}_${MODEL}.json" \
    --predictions_out  "$RESULTS/${TISSUE}_${MODEL}.test_predictions.csv"
done

echo "Done. Metrics: $RESULTS/${TISSUE}_{logreg,linear_svm,rbf_svm}.json"
