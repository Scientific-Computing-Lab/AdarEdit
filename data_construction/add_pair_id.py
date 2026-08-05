#!/usr/bin/env python3
"""Add a stable integer `pair_id` column to every human metadata file.

`pair_prefix` is the first 24 nt of the duplex. It is convenient to read but it
is **not** a unique key: *Alu* elements share a conserved 5' end, so 55 of the
884 substrates collide on that prefix. Auditing split disjointness on
`pair_prefix` therefore reports overlaps that do not exist.

`pair_id` is the substrate identifier: one integer per distinct `L + "A" + R`
duplex sequence, assigned by sorting the 884 substrates and numbering them from
1. The mapping is deterministic, so re-running this script reproduces the same
ids, and it is shared across tissues -- a substrate carries the same `pair_id`
wherever it appears.

Usage:
    python add_pair_id.py                # rewrite the metadata files in place
    python add_pair_id.py --check        # report only, write nothing
    python add_pair_id.py --out DIR      # write to DIR/<Tissue>/ instead
"""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HUMAN = REPO / "data" / "human"
TISSUES = ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]
SPLITS = ["train", "valid", "test"]


def substrate(line):
    """The full duplex sequence; identical for every site of the same Alu pair."""
    content = json.loads(line)["messages"][1]["content"]
    left = content.split("L:", 1)[1].split(", A:", 1)[0]
    right = content.split(", A:A, R:", 1)[1].split(", Alu Vienna Structure:", 1)[0]
    return left + "A" + right


def build_id_map():
    seqs = set()
    for tissue in TISSUES:
        for split in SPLITS:
            with open(HUMAN / tissue / f"{split}.jsonl") as fh:
                for line in fh:
                    seqs.add(substrate(line))
    return {s: i for i, s in enumerate(sorted(seqs), start=1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None,
                    help="write to this directory instead of alongside the data")
    args = ap.parse_args()

    id_map = build_id_map()
    print(f"{len(id_map)} distinct substrates -> pair_id 1..{len(id_map)}")

    prefixes = {s[:24] for s in id_map}
    print(f"  (for comparison, distinct 24-nt pair_prefix values: {len(prefixes)}; "
          f"{len(id_map) - len(prefixes)} substrates collide on the prefix)")

    for tissue in TISSUES:
        for split in SPLITS:
            jsonl = HUMAN / tissue / f"{split}.jsonl"
            meta = HUMAN / tissue / f"{split}.metadata.csv"
            ids = [id_map[substrate(l)] for l in open(jsonl)]
            with open(meta) as fh:
                rows = list(csv.DictReader(fh))
            if len(rows) != len(ids):
                raise SystemExit(f"{meta}: {len(rows)} metadata rows vs {len(ids)} jsonl records")
            if "pair_id" in rows[0]:
                print(f"  {tissue}/{split}: pair_id already present, skipping")
                continue
            for r, pid in zip(rows, ids):
                r["pair_id"] = pid
            if args.check:
                continue
            outdir = Path(args.out) / tissue if args.out else meta.parent
            outdir.mkdir(parents=True, exist_ok=True)
            fields = ["pair_id"] + [f for f in rows[0] if f != "pair_id"]
            with open(outdir / meta.name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader(); w.writerows(rows)
            print(f"  {tissue}/{split}: {len(rows)} rows -> {outdir / meta.name}")


if __name__ == "__main__":
    main()
