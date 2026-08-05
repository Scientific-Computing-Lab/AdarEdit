#!/usr/bin/env python3
"""Verify pair- and site-disjoint human splits before joint-model training."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

TISSUES = ["Artery", "Brain", "Combined", "Liver", "MuscleSkeletal"]
SPLITS = ["train", "valid", "test"]


def parse_site(line: str) -> tuple[str, str, int]:
    record = json.loads(line)
    user = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "user"
    )
    fields = {}
    for item in user.split(", "):
        if ":" in item:
            key, value = item.split(":", 1)
            fields[key] = value
    left = fields.get("L", "")
    sequence = (left + fields.get("A", "") + fields.get("R", "")).upper()
    sequence = sequence.replace("T", "U")
    structure = fields.get("Alu Vienna Structure", "")
    if not sequence or len(sequence) != len(structure):
        raise ValueError("Malformed sequence/structure record")
    return sequence, structure, len(left)


def read_pair_ids(metadata_path: Path) -> list[str]:
    with metadata_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "pair_id" not in rows[0]:
        raise ValueError(f"Missing pair_id column: {metadata_path}")
    return [row["pair_id"] for row in rows]


def main() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    data_root = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else repository_root / "data/human"
    )

    sites_by_split = {split: set() for split in SPLITS}
    pairs_by_split = {split: set() for split in SPLITS}
    record_counts = defaultdict(int)

    for tissue in TISSUES:
        for split in SPLITS:
            jsonl_path = data_root / tissue / f"{split}.jsonl"
            metadata_path = data_root / tissue / f"{split}.metadata.csv"
            if not jsonl_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Missing input pair: {jsonl_path}, {metadata_path}"
                )
            with jsonl_path.open() as handle:
                sites = [parse_site(line) for line in handle if line.strip()]
            pair_ids = read_pair_ids(metadata_path)
            if len(sites) != len(pair_ids):
                raise RuntimeError(
                    f"JSONL/metadata row mismatch for {tissue}/{split}: "
                    f"{len(sites)} != {len(pair_ids)}"
                )
            sites_by_split[split].update(sites)
            pairs_by_split[split].update(pair_ids)
            record_counts[(tissue, split)] = len(sites)

    failures = []
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        pair_overlap = pairs_by_split[left] & pairs_by_split[right]
        site_overlap = sites_by_split[left] & sites_by_split[right]
        print(
            f"{left}/{right}: pair_overlap={len(pair_overlap)} "
            f"site_overlap={len(site_overlap)}"
        )
        if pair_overlap:
            failures.append(f"{left}/{right} pair overlap: {len(pair_overlap)}")
        if site_overlap:
            failures.append(f"{left}/{right} site overlap: {len(site_overlap)}")

    print(
        "unique pairs: "
        + " ".join(
            f"{split}={len(pairs_by_split[split])}" for split in SPLITS
        )
    )
    print(
        "unique sites: "
        + " ".join(
            f"{split}={len(sites_by_split[split])}" for split in SPLITS
        )
    )
    print(
        "records by tissue/split: "
        + ", ".join(
            f"{tissue}/{split}={record_counts[(tissue, split)]}"
            for tissue in TISSUES
            for split in SPLITS
        )
    )

    if failures:
        raise RuntimeError("; ".join(failures))
    print("PASS: the human benchmark is pair-disjoint and site-disjoint.")


if __name__ == "__main__":
    main()
