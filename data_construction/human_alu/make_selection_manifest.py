#!/usr/bin/env python3
"""Generate the site-selection manifest from authoritative JSONL artifacts.

Use ``--within-tissue-root`` with the within-tissue train/valid dataset tree.
``--data-root`` is also supported for auditing a globally partitioned data tree.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


TISSUES = ("Artery", "Brain", "Liver", "MuscleSkeletal", "Combined")
SPLITS = ("train", "valid", "test")
WITHIN_TISSUE_SOURCES = {
    "Artery": ("Artery_Tibial/combine_1_1", "ArteryTibial"),
    "Brain": ("Brain_Cerebellum/combine_2_2", "BrainCerebellum"),
    "Liver": ("Liver/combine_3_3", "Liver"),
    "MuscleSkeletal": (
        "Muscle_Skeletal/combine_4_4",
        "MuscleSkeletal",
    ),
    "Combined": ("Combined/combine_5_5", "Combined"),
}


def parse_record(line: str):
    record = json.loads(line)
    user = next(x["content"] for x in record["messages"] if x["role"] == "user")
    label = next(
        x["content"].strip().lower()
        for x in record["messages"]
        if x["role"] == "assistant"
    )
    left = user.split("L:", 1)[1].split(", A:", 1)[0]
    right = user.split(", A:A, R:", 1)[1].split(
        ", Alu Vienna Structure:", 1
    )[0]
    structure = user.split(", Alu Vienna Structure:", 1)[1]
    return structure, left, right, label


def fingerprint(fields) -> str:
    payload = json.dumps(
        fields, ensure_ascii=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--data-root",
        type=Path,
        help="global data tree: <root>/<Tissue>/{train,valid,test}.jsonl",
    )
    sources.add_argument(
        "--within-tissue-root",
        type=Path,
        help="source dataset tree containing the five combine directories",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["tissue", "record_sha256", "count"])
        for tissue in TISSUES:
            counts = Counter()
            if args.data_root is not None:
                paths = [
                    args.data_root / tissue / f"{split}.jsonl"
                    for split in SPLITS
                ]
            else:
                relative, stem = WITHIN_TISSUE_SOURCES[tissue]
                paths = [
                    args.within_tissue_root / relative / f"{stem}_{split}.jsonl"
                    for split in ("train", "valid")
                ]
            for path in paths:
                with path.open() as source:
                    for line in source:
                        counts[fingerprint(parse_record(line))] += 1
            for record_sha256, count in sorted(counts.items()):
                writer.writerow([tissue, record_sha256, count])

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
