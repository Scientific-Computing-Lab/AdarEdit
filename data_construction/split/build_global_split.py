#!/usr/bin/env python3
"""
Build a SINGLE GLOBAL pair-disjoint train/valid/test split and apply it to all
five human tissues, so that a full 5x5 cross-tissue transfer matrix is
automatically pair-disjoint across every train->test combination.

Stage-one feasibility check only: this re-partitions the existing balanced
per-tissue site tables by a global pair->split map and audits
  (1) cross-tissue pair leakage (must be exactly 0 by construction), and
  (2) whether every tissue x split still has enough labelled, reasonably
      balanced sites to train/evaluate (esp. Muscle Skeletal).
No model training here.
"""
import os
import csv, os, json, random
from collections import defaultdict

# Per-tissue labelled-site tables (columns incl. pair_id) that predate the split.
# These are the upstream tables produced by the human_alu/ step; set to your own path.
SRC = os.environ.get("ADAREDIT_SITE_TABLES", "site_tables")
OUT = os.environ.get("ADAREDIT_SPLIT_OUT", ".")
TISSUES = ["Artery", "Brain", "Combined", "Liver", "MuscleSkeletal"]
FRACS = {"train": 0.64, "valid": 0.16, "test": 0.20}
SEED = 42

def load_tissue_sites(t):
    """All labelled sites for a tissue (pooled across the source splits)."""
    sites = []
    for sp in ("train", "valid", "test"):
        f = os.path.join(SRC, t, f"{sp}.metadata.csv")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            for row in csv.DictReader(fh):
                sites.append({"pair_id": row["pair_id"],
                              "yes": 1 if row["yes_no"] in ("yes", "1", "True") else 0})
    return sites

# 1) universe of pairs (union over all tissues)
tissue_sites = {t: load_tissue_sites(t) for t in TISSUES}
universe = sorted({s["pair_id"] for t in TISSUES for s in tissue_sites[t]})
print(f"universe of unique Alu pairs (union over tissues): {len(universe)}")

# 2) GLOBAL pair -> split assignment (one decision per pair, used everywhere)
rng = random.Random(SEED)
pairs = universe[:]
rng.shuffle(pairs)
n = len(pairs)
n_tr = int(round(FRACS["train"] * n))
n_va = int(round(FRACS["valid"] * n))
global_split = {}
for i, p in enumerate(pairs):
    global_split[p] = "train" if i < n_tr else ("valid" if i < n_tr + n_va else "test")
from collections import Counter
print("global pair split:", dict(Counter(global_split.values())))

# 3) re-partition each tissue's sites by the global pair map
def split_counts(t):
    c = {sp: {"sites": 0, "yes": 0, "no": 0, "pairs": set()} for sp in FRACS}
    for s in tissue_sites[t]:
        sp = global_split[s["pair_id"]]
        c[sp]["sites"] += 1
        c[sp]["yes" if s["yes"] else "no"] += 1
        c[sp]["pairs"].add(s["pair_id"])
    return c

per_tissue = {t: split_counts(t) for t in TISSUES}

# 4a) AUDIT cross-tissue pair leakage: test pairs of B vs train/valid pairs of A
def pairset(t, sp):
    return per_tissue[t][sp]["pairs"]

max_leak = 0
for B in TISSUES:                       # tissue we TEST on
    testB = pairset(B, "test")
    for A in TISSUES:                    # tissue we TRAIN on
        trainvalA = pairset(A, "train") | pairset(A, "valid")
        leak = len(testB & trainvalA)
        max_leak = max(max_leak, leak)
print(f"\n[AUDIT] max cross-tissue pair overlap (any test_B vs train/valid_A): {max_leak}")
print("        (0 = every train->test cell in the 5x5 matrix is pair-disjoint)")

# 4b) sufficiency / balance per tissue x split
print("\n[SUFFICIENCY] sites (yes/no) and pairs per tissue x split")
print(f"{'tissue':16s} {'split':6s} {'sites':>7s} {'yes':>6s} {'no':>6s} {'pairs':>6s} {'bal%':>6s}")
rows_out = []
for t in TISSUES:
    for sp in ("train", "valid", "test"):
        c = per_tissue[t][sp]
        bal = 100.0 * c["yes"] / c["sites"] if c["sites"] else 0
        print(f"{t:16s} {sp:6s} {c['sites']:7d} {c['yes']:6d} {c['no']:6d} {len(c['pairs']):6d} {bal:6.1f}")
        rows_out.append({"tissue": t, "split": sp, "sites": c["sites"],
                         "yes": c["yes"], "no": c["no"], "pairs": len(c["pairs"]),
                         "yes_frac": round(bal/100, 4)})

# persist the global map + the audit table
with open(os.path.join(OUT, "global_pair_split.json"), "w") as f:
    json.dump(global_split, f)
with open(os.path.join(OUT, "split_sufficiency_audit.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader(); w.writerows(rows_out)

# verdict
min_test = min(per_tissue[t]["test"]["sites"] for t in TISSUES)
print(f"\nsmallest tissue test set: {min_test} sites")
print("VERDICT:", "FEASIBLE — clean global split, 0 cross-tissue leakage"
      if max_leak == 0 and min_test >= 300 else "CHECK FAILED")
