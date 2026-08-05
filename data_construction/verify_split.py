#!/usr/bin/env python3
"""
Verify the pair/site-disjoint properties of the shipped human benchmark.

The split assignment is shipped as data (the `split` column of every
`data/human/<tissue>/*.metadata.csv`). This script re-derives every property
the manuscript claims about it, directly from the shipped JSONL, and fails
loudly if any claim does not hold. It trains nothing and needs no external input.

Checks
  1. Substrate disjointness  -- no Alu substrate appears in two different splits.
  2. Cross-tissue overlap    -- for every ordered tissue pair (A, B), no site in
                                A's train/valid appears in B's test. This is the
                                reported cross-tissue overlap criterion.
  3. Split proportions       -- pairs are assigned 64:16:20 at the substrate level.
  4. Substrate inventory     -- 884 unique substrates, every one graph-buildable
                                (dot-bracket length equals sequence length).
  5. Per-split class balance -- reported, not asserted: balancing was applied to
                                each dataset as a whole before splitting, so the
                                positive rate varies between splits.

Usage:
    python verify_split.py                 # from data_construction/
    python verify_split.py --data DIR      # explicit path to data/human
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

TISSUES = ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]
SPLITS = ["train", "valid", "test"]
EXPECTED_FRACS = {"train": 0.64, "valid": 0.16, "test": 0.20}
EXPECTED_SUBSTRATES = 884
TOL = 0.02          # allowed deviation on the 64:16:20 proportions


def parse(line):
    d = json.loads(line)
    c = d["messages"][1]["content"]
    L = c.split("L:", 1)[1].split(", A:", 1)[0]
    R = c.split(", A:A, R:", 1)[1].split(", Alu Vienna Structure:", 1)[0]
    struct = c.split("Alu Vienna Structure:", 1)[1] if "Alu Vienna Structure:" in c else ""
    label = 1 if d["messages"][2]["content"].strip() == "yes" else 0
    return (L, R), L + "A" + R, struct, label


def main(data_dir):
    data_dir = Path(data_dir)
    # site -> tissue -> {splits};  substrate -> {splits}
    site_map = defaultdict(lambda: defaultdict(set))
    sub_split = defaultdict(set)
    sub_struct = {}
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # tissue -> split -> [n, n_pos]

    for t in TISSUES:
        for s in SPLITS:
            f = data_dir / t / f"{s}.jsonl"
            if not f.exists():
                raise SystemExit(f"missing {f}")
            for line in open(f):
                site, sub, struct, label = parse(line)
                site_map[site][t].add(s)
                sub_split[sub].add(s)
                sub_struct.setdefault(sub, struct)
                counts[t][s][0] += 1
                counts[t][s][1] += label

    failures = []

    # 1. substrate disjointness
    multi = {k: v for k, v in sub_split.items() if len(v) > 1}
    print(f"[1] substrate disjointness      : {len(sub_split)} substrates, "
          f"{len(multi)} in more than one split")
    if multi:
        failures.append(f"{len(multi)} substrates span multiple splits")

    # 2. cross-tissue overlap (A train/valid -> B test)
    worst = 0
    for a in TISSUES:
        for b in TISSUES:
            n = sum(1 for tv in site_map.values()
                    if a in tv and b in tv
                    and (tv[a] & {"train", "valid"}) and "test" in tv[b])
            worst = max(worst, n)
    print(f"[2] cross-tissue site overlap   : worst A(train|valid)->B(test) cell = {worst} "
          f"(over {len(TISSUES)**2} ordered tissue pairs)")
    if worst:
        failures.append(f"cross-tissue overlap: {worst} sites")

    # 3. proportions at the substrate level
    c = Counter(next(iter(v)) for v in sub_split.values() if len(v) == 1)
    tot = sum(c.values())
    print(f"[3] split proportions           : " + "  ".join(
        f"{s}={c[s]} ({100*c[s]/tot:.1f}%, target {100*EXPECTED_FRACS[s]:.0f}%)" for s in SPLITS))
    for s in SPLITS:
        if abs(c[s] / tot - EXPECTED_FRACS[s]) > TOL:
            failures.append(f"{s} proportion {c[s]/tot:.3f} deviates from {EXPECTED_FRACS[s]}")

    # 4. inventory + graph-buildability
    bad = [k for k, st in sub_struct.items() if len(st) != len(k)]
    print(f"[4] substrate inventory         : {len(sub_split)} unique "
          f"(expected {EXPECTED_SUBSTRATES}), {len(bad)} with length mismatch")
    if len(sub_split) != EXPECTED_SUBSTRATES:
        failures.append(f"{len(sub_split)} substrates, expected {EXPECTED_SUBSTRATES}")
    if bad:
        failures.append(f"{len(bad)} substrates are not graph-buildable")

    # 5. class balance -- reported, not asserted
    print("[5] positive rate per split     : (balancing is dataset-level, so these vary)")
    lo = hi = None
    for t in TISSUES:
        cells = []
        for s in SPLITS:
            n, p = counts[t][s]
            frac = p / n
            lo = frac if lo is None else min(lo, frac)
            hi = frac if hi is None else max(hi, frac)
            cells.append(f"{s} {100*frac:.1f}%")
        print(f"      {t:16} " + "  ".join(cells))
    print(f"      range across all splits: {100*lo:.1f}%--{100*hi:.1f}% positive")

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("ALL CHECKS PASSED - the shipped split is substrate-disjoint, has zero "
          "cross-tissue site overlap, and is partitioned 64:16:20.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data/human"),
                   help="path to data/human")
    main(p.parse_args().data)
