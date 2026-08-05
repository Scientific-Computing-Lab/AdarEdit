#!/usr/bin/env python3
"""Build deterministic region-disjoint species train/valid/test datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SPLITS = ("train", "valid", "test")
DEFAULT_FRACTIONS = (0.64, 0.16, 0.20)
SYSTEM_MESSAGE = (
    "Predict whether the central adenosine (A) in the given RNA duplex "
    "will be edited to inosine (I) by an ADAR enzyme."
)
REQUIRED_COLUMNS = {
    "L",
    "R",
    "mfe_struct",
    "EditingClass",
    "region_id",
    "source_row_id",
    "focal_base",
}


def parse_fractions(value: str) -> tuple[float, float, float]:
    try:
        fractions = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fractions must be numeric") from exc
    if len(fractions) != 3 or any(item <= 0 for item in fractions):
        raise argparse.ArgumentTypeError("provide three positive fractions")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-9):
        raise argparse.ArgumentTypeError("fractions must sum to 1")
    return fractions  # type: ignore[return-value]


def stable_region_order(species: str, regions: Iterable[str], seed: int) -> list[str]:
    def key(region: str) -> str:
        payload = f"{seed}\0{species}\0{region}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return sorted(regions, key=key)


def split_counts(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError(f"at least three regions are required, found {n}")
    raw = [fraction * n for fraction in fractions]
    counts = [math.floor(value) for value in raw]
    remainder = n - sum(counts)
    order = sorted(range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    if any(count == 0 for count in counts):
        raise ValueError(f"split would contain no regions: n={n}, counts={counts}")
    return counts[0], counts[1], counts[2]


def read_balanced_sites(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def validate_row(path: Path, row: dict[str, str], row_number: int) -> None:
    label = row["EditingClass"].strip().lower()
    if label not in {"yes", "no"}:
        raise ValueError(f"{path}:{row_number}: invalid label {label!r}")
    if row["focal_base"].strip().upper() != "A":
        raise ValueError(f"{path}:{row_number}: focal_base is not A")
    sequence = f"{row['L']}A{row['R']}"
    structure = row["mfe_struct"]
    if len(sequence) != len(structure):
        raise ValueError(
            f"{path}:{row_number}: sequence length {len(sequence)} != "
            f"structure length {len(structure)}"
        )
    if not row["region_id"].strip():
        raise ValueError(f"{path}:{row_number}: empty region_id")


def make_record(row: dict[str, str]) -> dict[str, object]:
    content = (
        f"L:{row['L']}, A:A, R:{row['R']}, "
        f"Alu Vienna Structure:{row['mfe_struct']}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": content},
            {"role": "assistant", "content": row["EditingClass"].strip().lower()},
        ]
    }


def ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory must be empty or absent: {path}")
    path.mkdir(parents=True, exist_ok=True)


def build_species(
    species: str,
    input_path: Path,
    output_root: Path,
    seed: int,
    fractions: tuple[float, float, float],
) -> dict[str, object]:
    rows = read_balanced_sites(input_path)
    for row_number, row in enumerate(rows, start=2):
        validate_row(input_path, row, row_number)

    by_region: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_region[row["region_id"]].append(row)

    ordered_regions = stable_region_order(species, by_region, seed)
    train_n, valid_n, test_n = split_counts(len(ordered_regions), fractions)
    region_to_split = {
        region: (
            "train"
            if index < train_n
            else "valid"
            if index < train_n + valid_n
            else "test"
        )
        for index, region in enumerate(ordered_regions)
    }

    species_dir = output_root / species
    species_dir.mkdir(parents=True, exist_ok=False)
    json_handles = {
        split: (species_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in SPLITS
    }
    metadata_handles = {
        split: (species_dir / f"{split}.metadata.csv").open(
            "w", newline="", encoding="utf-8"
        )
        for split in SPLITS
    }
    metadata_fields = [
        "species",
        "region_id",
        "source_row_id",
        "label",
        "split",
        "Chr",
        "Position",
        "Strand",
        "Local_Position",
        "EditingLevel",
        "source_file",
    ]
    writers = {
        split: csv.DictWriter(metadata_handles[split], fieldnames=metadata_fields)
        for split in SPLITS
    }
    for writer in writers.values():
        writer.writeheader()

    counts = {split: Counter() for split in SPLITS}
    split_regions = {split: set() for split in SPLITS}
    try:
        for region in ordered_regions:
            split = region_to_split[region]
            split_regions[split].add(region)
            for row in sorted(
                by_region[region],
                key=lambda item: (
                    item.get("Position", ""),
                    int(item["source_row_id"]),
                ),
            ):
                label = row["EditingClass"].strip().lower()
                json_handles[split].write(json.dumps(make_record(row)) + "\n")
                writers[split].writerow(
                    {
                        "species": species,
                        "region_id": region,
                        "source_row_id": row["source_row_id"],
                        "label": label,
                        "split": split,
                        "Chr": row.get("Chr", ""),
                        "Position": row.get("Position", ""),
                        "Strand": row.get("Strand", ""),
                        "Local_Position": row.get("Local_Position", ""),
                        "EditingLevel": row.get("EditingLevel", ""),
                        "source_file": input_path.name,
                    }
                )
                counts[split][label] += 1
    finally:
        for handle in (*json_handles.values(), *metadata_handles.values()):
            handle.close()

    overlap = (
        (split_regions["train"] & split_regions["valid"])
        | (split_regions["train"] & split_regions["test"])
        | (split_regions["valid"] & split_regions["test"])
    )
    if overlap:
        raise RuntimeError(f"{species}: region leakage detected: {sorted(overlap)[:5]}")
    for split in SPLITS:
        if counts[split]["yes"] == 0 or counts[split]["no"] == 0:
            raise RuntimeError(f"{species}: {split} is missing one class: {counts[split]}")

    return {
        "species": species,
        "input": input_path.name,
        "seed": seed,
        "fractions": dict(zip(SPLITS, fractions)),
        "total_sites": len(rows),
        "total_regions": len(ordered_regions),
        "region_overlap_count": len(overlap),
        "splits": {
            split: {
                "sites": sum(counts[split].values()),
                "yes": counts[split]["yes"],
                "no": counts[split]["no"],
                "regions": len(split_regions[split]),
            }
            for split in SPLITS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--balanced-root",
        type=Path,
        required=True,
        help="Directory produced by prepare_balanced_ml_sets.R",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fractions", type=parse_fractions, default=DEFAULT_FRACTIONS,
        help="train,valid,test fractions [default: 0.64,0.16,0.20]",
    )
    args = parser.parse_args()

    ensure_empty_output(args.out_dir)
    inputs = sorted(args.balanced_root.glob("*/balanced_sites.csv"))
    if not inputs:
        raise SystemExit(
            f"No <species>/balanced_sites.csv inputs found under {args.balanced_root}"
        )

    summaries = []
    for input_path in inputs:
        species = input_path.parent.name
        summary = build_species(
            species, input_path, args.out_dir, args.seed, args.fractions
        )
        summaries.append(summary)
        split_text = ", ".join(
            f"{split}={summary['splits'][split]['sites']}"
            for split in SPLITS
        )
        print(f"[{species}] {split_text}; regions={summary['total_regions']}")

    summary_path = args.out_dir / "species_split_summary.json"
    summary_path.write_text(json.dumps({"species": summaries}, indent=2) + "\n")
    print(f"PASS: region-disjoint species benchmark written to {args.out_dir}")


if __name__ == "__main__":
    main()
