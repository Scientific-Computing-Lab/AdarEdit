#!/usr/bin/env bash
set -euo pipefail

# End-to-end human benchmark construction.
#
# Usage:
#   bash data_construction/run_human_construction.sh \
#       data/raw/editing_levels \
#       /path/to/new_output_directory \
#       /path/to/temporary_work_directory
#
# The output directory must not already contain files. Published data/human is
# used only as an optional equality reference and is never modified.

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 RAW_EDITING_DIR OUTPUT_DIR [WORK_DIR]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="$1"
OUTPUT_DIR="$2"
WORK_DIR="${3:-${TMPDIR:-/tmp}/adaredit_human_construction_${USER:-user}_$$}"
BALANCED_DIR="${WORK_DIR}/published_site_pools"

mkdir -p "${WORK_DIR}"

echo "[1/3] Selecting the exact published labelled records from the GTEx tables"
python "${SCRIPT_DIR}/human_alu/select_published_site_pool.py" \
  --data-dir "${RAW_DIR}" \
  --output-dir "${BALANCED_DIR}" \
  --yes-cutoff 15 \
  --no-cutoff 1 \
  --manifest "${SCRIPT_DIR}/human_alu/published_site_selection.tsv.gz"

echo "[2/3] Applying the canonical global pair-disjoint split"
python "${SCRIPT_DIR}/split/build_human_global_split.py" \
  --balanced-dir "${BALANCED_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --split-map "${SCRIPT_DIR}/split/global_pair_split.json" \
  --invalid-policy drop \
  --expected-substrates 884 \
  --canonical-reference "${REPO_DIR}/data/human"

echo "[3/3] Verifying the reconstructed split"
python "${SCRIPT_DIR}/verify_split.py" --data "${OUTPUT_DIR}"

echo "PASS: reconstructed human benchmark: ${OUTPUT_DIR}"
echo "Temporary exact published site pools: ${BALANCED_DIR}"
