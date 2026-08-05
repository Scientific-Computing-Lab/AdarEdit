#!/usr/bin/env python3
"""Validate JSONL/metadata alignment and region disjointness for species data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


SPLITS = ("train", "valid", "test")


def parse_user_content(content: str) -> tuple[str, str, str]:
    prefix = "L:"
    middle = ", A:A, R:"
    structure_marker = ", Alu Vienna Structure:"
    if not content.startswith(prefix) or middle not in content or structure_marker not in content:
        raise ValueError("unexpected user-message format")
    left, remainder = content[len(prefix):].split(middle, 1)
    right, structure = remainder.split(structure_marker, 1)
    return left, right, structure


def validate_species(species_dir: Path) -> None:
    regions_by_split: dict[str, set[str]] = {}
    total_sites = 0
    for split in SPLITS:
        jsonl_path = species_dir / f"{split}.jsonl"
        metadata_path = species_dir / f"{split}.metadata.csv"
        if not jsonl_path.exists() or not metadata_path.exists():
            raise RuntimeError(f"{species_dir.name}: missing {split} JSONL or metadata")

        json_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            metadata = list(csv.DictReader(handle))
        if len(json_lines) != len(metadata):
            raise RuntimeError(
                f"{species_dir.name}/{split}: {len(json_lines)} JSONL rows != "
                f"{len(metadata)} metadata rows"
            )

        labels = Counter()
        for line_number, (line, meta) in enumerate(zip(json_lines, metadata), start=1):
            record = json.loads(line)
            messages = record.get("messages", [])
            user = next(message["content"] for message in messages if message["role"] == "user")
            label = next(
                message["content"].strip().lower()
                for message in messages
                if message["role"] == "assistant"
            )
            left, right, structure = parse_user_content(user)
            if len(left) + 1 + len(right) != len(structure):
                raise RuntimeError(
                    f"{species_dir.name}/{split}:{line_number}: length mismatch"
                )
            if label not in {"yes", "no"} or label != meta["label"]:
                raise RuntimeError(
                    f"{species_dir.name}/{split}:{line_number}: label mismatch"
                )
            if meta["split"] != split:
                raise RuntimeError(
                    f"{species_dir.name}/{split}:{line_number}: metadata split mismatch"
                )
            labels[label] += 1

        if labels["yes"] == 0 or labels["no"] == 0:
            raise RuntimeError(f"{species_dir.name}/{split}: missing class: {labels}")
        regions_by_split[split] = {row["region_id"] for row in metadata}
        total_sites += len(json_lines)
        print(
            f"[{species_dir.name}] {split}: sites={len(json_lines)}, "
            f"yes={labels['yes']}, no={labels['no']}, "
            f"regions={len(regions_by_split[split])}"
        )

    overlap = (
        (regions_by_split["train"] & regions_by_split["valid"])
        | (regions_by_split["train"] & regions_by_split["test"])
        | (regions_by_split["valid"] & regions_by_split["test"])
    )
    if overlap:
        raise RuntimeError(
            f"{species_dir.name}: {len(overlap)} regions occur in multiple splits"
        )
    print(f"[{species_dir.name}] PASS: {total_sites} sites; region overlap=0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Root containing <species>/{train,valid,test}.jsonl and metadata",
    )
    args = parser.parse_args()
    species_dirs = sorted(path for path in args.data_root.iterdir() if path.is_dir())
    if not species_dirs:
        raise SystemExit(f"No species directories found under {args.data_root}")
    for species_dir in species_dirs:
        validate_species(species_dir)
    print("ALL SPECIES SPLIT CHECKS PASSED")


if __name__ == "__main__":
    main()
