#!/usr/bin/env bash
set -euo pipefail

ANALYSIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$ANALYSIS_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MPL_DIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/adaredit_species_sensitivity_mpl_${USER:-user}}"

mkdir -p "$MPL_DIR"
export MPLCONFIGDIR="$MPL_DIR"

cd "$REPO_DIR"
"$PYTHON_BIN" analyses/species_sensitivity/scripts/compute_full_dataset_distances.py \
  --species "Octopus bimaculoides=$REPO_DIR/data/species_prebalancing/Octopus_prebalancing.csv.gz" \
  --species "Ptychodera flava=$REPO_DIR/data/species_prebalancing/Ptychodera_prebalancing.csv.gz" \
  --species "Strongylocentrotus purpuratus=$REPO_DIR/data/species_prebalancing/Strongylocentrotus_prebalancing.csv.gz" \
  --output "$REPO_DIR/analyses/species_sensitivity/raw_data/species_inter_site_distances.json"
"$PYTHON_BIN" analyses/species_sensitivity/scripts/sensitivity.py
"$PYTHON_BIN" analyses/species_sensitivity/scripts/make_species_figure.py
"$PYTHON_BIN" analyses/species_sensitivity/scripts/validate_outputs.py

echo "[Supplementary Fig. S5] PASS"
