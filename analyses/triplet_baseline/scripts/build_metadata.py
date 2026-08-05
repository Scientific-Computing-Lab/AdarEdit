#!/usr/bin/env python3
"""
Convert the global-split human jsonl into the `{split}.metadata.csv` format
consumed by `train_triplet_baseline.py` (columns: full_seq, structure, L, yes_no).

This produces the inputs for a faithful target-centered triplet-feature baseline
(logistic regression / linear SVM) that uses a target-centered +/-25-nt window over
Xue-style base + dot-bracket triplets, on the pair-disjoint human split.

Output goes to this analysis' own `work/<tissue>/` directory. It must NEVER be
written into `data/human/`: the canonical `{split}.metadata.csv` files there use a
different schema (pair_prefix, yes_no, split) and document the pair-disjoint split
provenance, so writing this schema over them destroys that record.

Usage:
  python build_metadata.py <tissue> [--out_dir DIR]   # e.g. Liver
"""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]          # analyses/triplet_baseline/
DATA = REPO / "data/human"                          # read-only
DEFAULT_OUT = HERE / "work"                         # never data/human


def parse(js):
    c = js["messages"][1]["content"]
    L = c.split("L:", 1)[1].split(", A:", 1)[0]
    A = c.split(", A:", 1)[1].split(", R:", 1)[0]
    R = c.split(", R:", 1)[1].split(", Alu", 1)[0] if ", R:" in c else ""
    struct = c.split("Structure:", 1)[1].strip()
    yn = js["messages"][2]["content"].strip().lower()
    return L, A, R, struct, yn


def main(tissue, out_root):
    out_root = Path(out_root).resolve()
    if out_root == DATA.resolve() or DATA.resolve() in out_root.parents:
        raise SystemExit(f"refusing to write into the canonical data directory: {out_root}")
    outdir = out_root / tissue
    outdir.mkdir(parents=True, exist_ok=True)
    for sp in ["train", "valid", "test"]:
        rows = []
        for line in open(DATA / tissue / f"{sp}.jsonl"):
            L, A, R, struct, yn = parse(json.loads(line))
            rows.append({"full_seq": L + A + R, "structure": struct, "L": L, "yes_no": yn})
        with open(outdir / f"{sp}.metadata.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["full_seq", "structure", "L", "yes_no"])
            w.writeheader(); w.writerows(rows)
        nmatch = sum(1 for r in rows if len(r["full_seq"]) == len(r["structure"]))
        print(f"{tissue} {sp}: rows={len(rows)} len-match(kept)={nmatch} -> {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tissue", nargs="?", default="Liver")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT),
                   help="where to write the triplet-baseline metadata "
                        "(default: analyses/triplet_baseline/work; data/human is refused)")
    a = p.parse_args()
    main(a.tissue, a.out_dir)
