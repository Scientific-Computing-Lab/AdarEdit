#!/usr/bin/env python3
"""Create row-aligned provenance metadata for model-ready species JSONL files."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import tempfile
from collections import defaultdict, deque
from pathlib import Path


SPLITS = ("train", "valid", "test")
METADATA_FIELDS = (
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
)


def parse_record(line: str) -> tuple[str, str, str, str]:
    record = json.loads(line)
    messages = record.get("messages", [])
    user = next(message["content"] for message in messages if message["role"] == "user")
    label = next(
        message["content"].strip().lower()
        for message in messages
        if message["role"] == "assistant"
    )
    prefix = "L:"
    middle = ", A:A, R:"
    structure_marker = ", Alu Vienna Structure:"
    if not user.startswith(prefix) or middle not in user or structure_marker not in user:
        raise ValueError("unexpected user-message format")
    left, remainder = user[len(prefix):].split(middle, 1)
    right, structure = remainder.split(structure_marker, 1)
    if label not in {"yes", "no"}:
        raise ValueError(f"unexpected label: {label!r}")
    return left, right, structure, label


def source_key(row: dict[str, str]) -> tuple[str, str, str, str] | None:
    try:
        local_position = int(float(row["Local_Position"]))
        editing_level = float(row["EditingLevel"])
    except (TypeError, ValueError):
        return None
    sequence = row["small_ds_seq"].upper()
    structure = row["mfe_struct"]
    if not 1 <= local_position <= len(sequence) or len(sequence) != len(structure):
        return None
    if editing_level > 0.15:
        label = "yes"
    elif editing_level < 0.001:
        label = "no"
    else:
        return None
    return (
        sequence[: local_position - 1],
        sequence[local_position:],
        structure,
        label,
    )


def load_source_rows(path: Path) -> dict[tuple[str, str, str, str], deque[dict[str, str]]]:
    indexed: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for source_row_id, row in enumerate(csv.DictReader(handle), start=1):
            key = source_key(row)
            if key is None:
                continue
            region_id = ":".join(
                (row["Chr"], row["Strand"], row["start_cluster"], row["end_cluster"])
            )
            indexed[key].append(
                {
                    "region_id": region_id,
                    "source_row_id": str(source_row_id),
                    "Chr": row["Chr"],
                    "Position": row["Position"],
                    "Strand": row["Strand"],
                    "Local_Position": row["Local_Position"],
                    "EditingLevel": row["EditingLevel"],
                }
            )

    queues: dict[tuple[str, str, str, str], deque[dict[str, str]]] = {}
    for key, rows in indexed.items():
        regions = {row["region_id"] for row in rows}
        if len(regions) != 1:
            raise RuntimeError(
                "a serialized record maps to source rows from multiple genomic regions"
            )
        queues[key] = deque(sorted(rows, key=lambda row: int(row["source_row_id"])))
    return queues


def write_metadata(path: Path, rows: list[dict[str, str]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"metadata already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_species_metadata(
    species_dir: Path,
    source_path: Path,
    overwrite: bool,
) -> int:
    source_rows = load_source_rows(source_path)
    assigned_regions: dict[str, str] = {}
    total = 0

    for split in SPLITS:
        jsonl_path = species_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(jsonl_path)
        metadata: list[dict[str, str]] = []
        for line_number, line in enumerate(
            jsonl_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            key = parse_record(line)
            candidates = source_rows.get(key)
            if not candidates:
                raise RuntimeError(
                    f"{species_dir.name}/{split}:{line_number}: no unused source match"
                )
            source = candidates.popleft()
            region_id = source["region_id"]
            prior_split = assigned_regions.setdefault(region_id, split)
            if prior_split != split:
                raise RuntimeError(
                    f"{species_dir.name}: region {region_id} occurs in both "
                    f"{prior_split} and {split}"
                )
            metadata.append(
                {
                    "species": species_dir.name,
                    "region_id": region_id,
                    "source_row_id": source["source_row_id"],
                    "label": key[3],
                    "split": split,
                    "Chr": source["Chr"],
                    "Position": source["Position"],
                    "Strand": source["Strand"],
                    "Local_Position": source["Local_Position"],
                    "EditingLevel": source["EditingLevel"],
                    "source_file": source_path.name,
                }
            )
        write_metadata(species_dir / f"{split}.metadata.csv", metadata, overwrite)
        total += len(metadata)
        print(f"[{species_dir.name}] {split}: wrote {len(metadata)} metadata rows")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing <species>/{train,valid,test}.jsonl",
    )
    parser.add_argument(
        "--prebalancing-root",
        type=Path,
        required=True,
        help="Directory containing <species>_prebalancing.csv.gz",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    species_dirs = sorted(path for path in args.data_root.iterdir() if path.is_dir())
    if not species_dirs:
        raise SystemExit(f"No species directories found under {args.data_root}")
    total = 0
    for species_dir in species_dirs:
        source_path = args.prebalancing_root / f"{species_dir.name}_prebalancing.csv.gz"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        total += build_species_metadata(species_dir, source_path, args.overwrite)
    print(f"PASS: wrote aligned metadata for {total} species records")


if __name__ == "__main__":
    main()
