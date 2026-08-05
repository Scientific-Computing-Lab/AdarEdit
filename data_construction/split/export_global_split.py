#!/usr/bin/env python3
"""
Export the hardened, GLOBALLY pair-disjoint datasets ready for training.

Output mirrors the shipped benchmark layout so the trainer can
consume it unchanged (point --split_root at export/human or export/species):

  export/human/<Tissue>/{train,valid,test}.jsonl          (+ .metadata.csv)
  export/species/<Species>/{train,valid,test}.jsonl       (+ .metadata.csv)

Human tissues use ONE global pair->split map (so the 5x5 cross-tissue matrix is
automatically pair-disjoint). Species use a per-species cluster(region)-disjoint
split. Records are copied verbatim; only their split bucket changes.
"""
import csv
import glob
import json
import os
import random

HSRC = os.environ.get("ADAREDIT_SITE_TABLES", "site_tables")   # human per-tissue tables
DS   = os.environ.get("ADAREDIT_SPECIES_CSV", "datasets")      # species per-cluster tables
OUT  = os.environ.get("ADAREDIT_SPLIT_OUT", "export")
# The authoritative pair->split map that reproduces the reported results, shipped
# next to this script (derived from the released metadata). build_global_split.py
# documents how the split was created, but rebuilding can differ slightly by
# RNG/rounding, so this shipped map is the ground truth. Override with
# ADAREDIT_SPLIT_MAP to point at a freshly built map instead.
_HERE = os.path.dirname(os.path.abspath(__file__))
MAP  = os.environ.get("ADAREDIT_SPLIT_MAP", os.path.join(_HERE, "global_pair_split.json"))
TISSUES = ["Artery", "Brain", "Combined", "Liver", "MuscleSkeletal"]
SPECIES = {"Octopus_Bimaculoides": "Octopus",
           "Ptychodera_Flava": "Ptychodera",
           "Strongylocentrotus_Purpuratus": "Strongylocentrotus"}
SEED = 42

def fields(content):
    d = {}
    for kv in content.split(", "):
        k = kv.split(":", 1)
        if len(k) == 2:
            d[k[0]] = k[1]
    return d

def write_split(out_dir, buckets, meta_buckets):
    os.makedirs(out_dir, exist_ok=True)
    for sp in ("train", "valid", "test"):
        with open(f"{out_dir}/{sp}.jsonl", "w") as f:
            for rec in buckets[sp]:
                f.write(json.dumps(rec) + "\n")
        with open(f"{out_dir}/{sp}.metadata.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pair_or_region", "yes_no", "split"])
            for m in meta_buckets[sp]:
                w.writerow(m)

# ---------------- HUMAN: one global pair->split map ----------------
gmap = json.load(open(MAP))
print("HUMAN (global pair-disjoint)")
for t in TISSUES:
    buckets = {"train": [], "valid": [], "test": []}
    metab = {"train": [], "valid": [], "test": []}
    for old in ("train", "valid", "test"):
        jl = f"{HSRC}/{t}/{old}.jsonl"; mc = f"{HSRC}/{t}/{old}.metadata.csv"
        if not os.path.exists(jl):
            continue
        with open(jl) as fj, open(mc) as fm:
            mr = csv.DictReader(fm)
            for line, row in zip(fj, mr):
                pid = row["pair_id"]
                sp = gmap[pid]
                buckets[sp].append(json.loads(line))
                metab[sp].append([pid, row["yes_no"], sp])
    write_split(f"{OUT}/human/{t}", buckets, metab)
    print(f"  {t:16s} train/valid/test = "
          f"{len(buckets['train'])}/{len(buckets['valid'])}/{len(buckets['test'])}")

# ---------------- SPECIES: per-species cluster(region)-disjoint ----------------
print("SPECIES (cluster/region-disjoint)")
for sp_dir, name in SPECIES.items():
    recs = []
    for f in glob.glob(f"{DS}/{sp_dir}/*/*train*.jsonl") + glob.glob(f"{DS}/{sp_dir}/*/*valid*.jsonl"):
        with open(f) as fh:
            for line in fh:
                rec = json.loads(line)
                um = [m for m in rec["messages"] if m["role"] == "user"][0]["content"]
                lbl = [m for m in rec["messages"] if m["role"] == "assistant"][0]["content"].strip().lower()
                region = fields(um).get("Alu Vienna Structure", "")
                recs.append((region, lbl, rec))
    regions = sorted({r for r, _, _ in recs})
    rng = random.Random(SEED); rng.shuffle(regions)
    n = len(regions); n_tr = int(round(.64*n)); n_va = int(round(.16*n))
    rsplit = {r: ("train" if i < n_tr else "valid" if i < n_tr+n_va else "test")
              for i, r in enumerate(regions)}
    buckets = {"train": [], "valid": [], "test": []}
    metab = {"train": [], "valid": [], "test": []}
    for region, lbl, rec in recs:
        s = rsplit[region]
        buckets[s].append(rec); metab[s].append([f"region_{regions.index(region)}", lbl, s])
    write_split(f"{OUT}/species/{name}", buckets, metab)
    print(f"  {name:18s} train/valid/test = "
          f"{len(buckets['train'])}/{len(buckets['valid'])}/{len(buckets['test'])}")

print(f"\nExported under: {OUT}")
