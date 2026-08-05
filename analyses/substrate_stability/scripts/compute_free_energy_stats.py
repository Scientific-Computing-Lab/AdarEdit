#!/usr/bin/env python3
"""
Thermodynamic-stability (free-energy) summary of the Alu-duplex substrates.

Quantifies how thermodynamically stable the substrates are, to accompany the
paired-fraction stability characterisation:
  - intrinsic substrate folds  (raw_data/substrate_stability_viennaRNA.csv):
        RNAfold minimum free energy (MFE) per substrate, folded in isolation.
  - genomic-context folds       (raw_data/context_folding_results.csv):
        MFE and ensemble free energy when each pair is re-folded with +-200 bp
        of real hg38 flanking sequence.

Writes: data/free_energy_stats.json
Run:    python scripts/compute_free_energy_stats.py
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]          # analyses/substrate_stability/
RAW  = HERE / "raw_data"
OUT  = HERE / "data"; OUT.mkdir(exist_ok=True)


def q(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return {"n": int(s.size), "median": float(s.median()),
            "q1": float(s.quantile(0.25)), "q3": float(s.quantile(0.75)),
            "min": float(s.min()), "max": float(s.max())}


# ---- intrinsic substrate folds (isolated) ----------------------------------
sub = pd.read_csv(RAW / "substrate_stability_viennaRNA.csv")
mfe_per_nt = (pd.to_numeric(sub["mfe_kcal_mol"], errors="coerce") /
              pd.to_numeric(sub["seq_len"], errors="coerce"))

intrinsic = {
    "seq_len_nt":        q(sub["seq_len"]),
    "paired_fraction":   q(sub["paired_frac"]),
    "mfe_kcal_mol":      q(sub["mfe_kcal_mol"]),
    "mfe_kcal_mol_per_nt": q(mfe_per_nt),
}

# ---- genomic-context folds (+-200 bp real flanks) --------------------------
ctx = pd.read_csv(RAW / "context_folding_results.csv")
# 100000 is ViennaRNA's failed-ensemble sentinel (211/865 rows); drop it before
# summarising the ensemble free energy, else it inflates the median/quartiles/max.
ens = pd.to_numeric(ctx["ensemble_free_energy"], errors="coerce")
ens = ens[ens < 99999]
context = {
    "ctx_len_nt":             q(ctx["ctx_len"]),
    "ctx_mfe_kcal_mol":       q(ctx["ctx_mfe_kcal"]),
    "ensemble_free_energy_kcal_mol": q(ens),
    "dsrna_fraction":         q(ctx["dsrna_frac"]),
}

summary = {
    "description": "Thermodynamic-stability (free-energy) summary of Alu-duplex substrates.",
    "intrinsic_isolated_fold": intrinsic,
    "genomic_context_fold": context,
    "headline": {
        "intrinsic_mfe_median_kcal_mol": round(intrinsic["mfe_kcal_mol"]["median"], 1),
        "intrinsic_mfe_per_nt_median_kcal_mol": round(intrinsic["mfe_kcal_mol_per_nt"]["median"], 3),
        "context_mfe_median_kcal_mol": round(context["ctx_mfe_kcal_mol"]["median"], 1),
    },
}

with open(OUT / "free_energy_stats.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"intrinsic substrates: n={intrinsic['mfe_kcal_mol']['n']}")
print(f"  MFE median = {intrinsic['mfe_kcal_mol']['median']:.1f} kcal/mol "
      f"(IQR {intrinsic['mfe_kcal_mol']['q1']:.0f} to {intrinsic['mfe_kcal_mol']['q3']:.0f})")
print(f"  MFE per nt median = {intrinsic['mfe_kcal_mol_per_nt']['median']:.3f} kcal/mol/nt")
print(f"  seq_len median = {intrinsic['seq_len_nt']['median']:.0f} nt, "
      f"paired_frac median = {intrinsic['paired_fraction']['median']:.3f}")
print(f"context MFE median = {context['ctx_mfe_kcal_mol']['median']:.1f} kcal/mol")
print("wrote", OUT / "free_energy_stats.json")
