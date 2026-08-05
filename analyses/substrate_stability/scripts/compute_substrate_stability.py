#!/usr/bin/env python3
"""
Per-substrate ViennaRNA folding of the Alu-duplex substrates (from-scratch step).

For each unique substrate (L + A + R sequence), computes with ViennaRNA:
  - seq_len        = substrate length (nt)
  - paired_frac    = fraction of paired positions in the fixed (input) structure
  - mfe_kcal_mol   = minimum free energy of the substrate (kcal/mol)
  - bp_distance    = base-pair distance between the fixed structure and the MFE structure
  - fixed_eq_mfe   = 1 if the fixed structure equals the MFE structure
  - mean_pair_prob = mean per-position pairing probability (partition function)
  - mean_entropy   = mean per-position pairing entropy

Reads all human substrates from the global pair-disjoint split (the substrate
sequence/structure is tissue-independent, so every split/tissue is scanned and
substrates are de-duplicated by their full sequence).

Output: raw_data/substrate_stability_viennaRNA.csv   (one row per unique substrate)
Run:    python scripts/compute_substrate_stability.py     (requires ViennaRNA)
"""
import csv
import json
import math
from pathlib import Path

try:
    import RNA
except ImportError:
    raise RuntimeError("ViennaRNA not available — install with: pip install ViennaRNA")

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]                 # analyses/substrate_stability/
INPUT_FILES = sorted((REPO / "data" / "human").glob("*/*.jsonl"))
OUT = HERE / "raw_data"; OUT.mkdir(exist_ok=True)


def parse_record(line):
    d = json.loads(line)
    content = d["messages"][1]["content"]
    parts = content.split(", A:A, R:")
    L = parts[0][2:]
    rest = parts[1] if len(parts) > 1 else ""
    R = rest.split(", Alu Vienna Structure:")[0]
    struct = content.split("Alu Vienna Structure:")[1] if "Alu Vienna Structure:" in content else ""
    return L + "A" + R, struct


def paired_fraction(struct):
    return sum(1 for c in struct if c in "()") / len(struct)


def compute_stability(seq, fixed_struct):
    """Full ViennaRNA stability computation for one substrate."""
    fc = RNA.fold_compound(seq)
    mfe_struct, mfe_energy = fc.mfe()
    bp_dist = RNA.bp_distance(fixed_struct, mfe_struct)

    fc.pf()
    bp_probs = fc.bpp()                     # n×n matrix of pair probabilities
    n = len(seq)
    pair_probs, entropies = [], []
    for i in range(1, n + 1):
        p_paired = sum(bp_probs[min(i, j)][max(i, j)]
                       for j in range(1, n + 1) if j != i
                       if bp_probs[min(i, j)][max(i, j)] > 0)
        p_paired = min(p_paired, 1.0)
        pair_probs.append(p_paired)
        p_u = 1.0 - p_paired
        h = 0.0
        if p_paired > 1e-12:
            h -= p_paired * math.log(p_paired)
        if p_u > 1e-12:
            h -= p_u * math.log(p_u)
        entropies.append(h)

    return {
        "seq_len":        n,
        "paired_frac":    round(paired_fraction(fixed_struct), 6),
        "mfe_kcal_mol":   round(float(mfe_energy), 4),
        "bp_distance":    int(bp_dist),
        "fixed_eq_mfe":   int(bp_dist == 0),
        "mean_pair_prob": round(sum(pair_probs) / n, 6),
        "mean_entropy":   round(sum(entropies) / n, 6),
    }


def main():
    print(f"Collecting substrates from {len(INPUT_FILES)} split files...")
    substrates = {}                         # full_seq -> fixed_struct
    for fpath in INPUT_FILES:
        with open(fpath) as f:
            for line in f:
                seq, struct = parse_record(line)
                if seq not in substrates and struct and len(struct) == len(seq):
                    substrates[seq] = struct
    n_total = len(substrates)
    print(f"  Unique substrates to process: {n_total}")

    fieldnames = ["substrate_idx", "seq_len", "paired_frac", "mfe_kcal_mol",
                  "bp_distance", "fixed_eq_mfe", "mean_pair_prob", "mean_entropy"]
    results = []
    for i, (seq, struct) in enumerate(substrates.items()):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{n_total}] seq_len={len(seq)}", flush=True)
        try:
            m = compute_stability(seq, struct)
            m["substrate_idx"] = i
            results.append(m)
        except Exception as e:
            print(f"  WARNING substrate {i}: {e}")

    out_path = OUT / "substrate_stability_viennaRNA.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    n_zero = sum(1 for r in results if r["bp_distance"] == 0)
    print(f"\nResults saved: {out_path}")
    print(f"  Total substrates: {len(results)}")
    print(f"  bp_distance = 0 (fixed = MFE): {n_zero} ({100*n_zero/len(results):.1f}%)")


if __name__ == "__main__":
    main()
