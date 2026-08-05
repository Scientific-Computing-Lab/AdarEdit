#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
    echo "Usage: bash data_construction/run_species_construction.sh MANIFEST.csv OUTPUT_DIR [extra options]" >&2
    exit 2
fi

manifest="$1"
output_dir="$2"
shift 2

python "${SCRIPT_DIR}/run_species_construction.py" \
    --manifest "${manifest}" \
    --out-dir "${output_dir}" \
    "$@"
