#!/usr/bin/env python3
"""Compute full-dataset negative distances for Supplementary Fig. S5a.

For each species, operational negatives (editing level < 0.1%) are measured
against benchmark-positive sites (editing level > 15%) in the same dsRNA
region and arm. Intermediate/boundary sites are neither negatives nor
reference positives. All negatives remain in the denominator, including those
whose region/arm contains no benchmark-positive site. Consecutive-positive
spacing is calculated from these same complete pre-balancing tables.

The compressed species tables are distributed with the repository. Supply them
as repeatable NAME=PATH arguments; the generated JSON is the source for panel a
and for the spacing statistics reported in its caption.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


THRESHOLDS = (5, 10, 20, 30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_species(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--species must have the form 'Species name=/path/to/input.csv'"
        )
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path).expanduser()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError(
            "--species must have the form 'Species name=/path/to/input.csv'"
        )
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"input file does not exist: {path}")
    return name, path


def region_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["Chr"],
        row["Strand"],
        row["start_1ds_genome"],
        row["end_1ds_genome"],
        row["start_2ds_genome"],
        row["end_2ds_genome"],
    )


def open_text(path: Path):
    """Open either a plain CSV or a gzip-compressed CSV as text."""
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", newline="")
    return path.open(newline="")


def analyze(
    name: str,
    path: Path,
    positive_threshold: float,
    negative_threshold: float,
) -> dict[str, object]:
    regions: dict[tuple[tuple[str, ...], int], list[tuple[int, float]]] = (
        defaultdict(list)
    )
    total_rows = 0
    parsed_rows = 0

    with open_text(path) as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            try:
                position = int(float(row["Position"]))
                editing_level = float(row["EditingLevel"])
                arm1_start = int(float(row["start_1ds_genome"]))
                arm1_end = int(float(row["end_1ds_genome"]))
            except (KeyError, TypeError, ValueError):
                continue
            arm = 1 if arm1_start <= position <= arm1_end else 2
            regions[(region_key(row), arm)].append((position, editing_level))
            parsed_rows += 1

    negative_count = 0
    positive_count = 0
    intermediate_count = 0
    distances: list[int] = []
    positive_spacings: list[int] = []
    negatives_without_positive = 0

    for sites in regions.values():
        positives = [
            position
            for position, level in sites
            if level > positive_threshold
        ]
        negatives = [
            position
            for position, level in sites
            if level < negative_threshold
        ]
        positive_count += len(positives)
        negative_count += len(negatives)
        intermediate_count += sum(
            negative_threshold <= level <= positive_threshold
            for _, level in sites
        )
        unique_positives = sorted(set(positives))
        positive_spacings.extend(
            right - left
            for left, right in zip(
                unique_positives[:-1], unique_positives[1:]
            )
        )

        if not positives:
            negatives_without_positive += len(negatives)
            continue
        distances.extend(
            min(abs(negative - positive) for positive in positives)
            for negative in negatives
        )

    cumulative: dict[int, int] = {
        threshold: sum(distance <= threshold for distance in distances)
        for threshold in THRESHOLDS
    }
    within_rows = [
        {
            "threshold_bp": threshold,
            "negatives_within": cumulative[threshold],
            "negatives_beyond_or_without_positive": (
                negative_count - cumulative[threshold]
            ),
            "frac_within_all_negatives": (
                cumulative[threshold] / negative_count if negative_count else 0.0
            ),
        }
        for threshold in THRESHOLDS
    ]

    bin_counts = {
        "≤5": cumulative[5],
        "6–10": cumulative[10] - cumulative[5],
        "11–20": cumulative[20] - cumulative[10],
        "21–30": cumulative[30] - cumulative[20],
        ">30 or no positive": negative_count - cumulative[30],
    }
    bins = [
        {
            "bin": label,
            "count": count,
            "fraction_all_negatives": count / negative_count
            if negative_count
            else 0.0,
        }
        for label, count in bin_counts.items()
    ]

    return {
        "species": name,
        "source_file": path.name,
        "source_sha256": sha256(path),
        "positive_definition": f"EditingLevel > {positive_threshold}",
        "negative_definition": f"EditingLevel < {negative_threshold}",
        "distance_definition": (
            "minimum absolute genomic distance to a benchmark-positive site "
            "in the same dsRNA region and arm"
        ),
        "positive_spacing_definition": (
            "distance between consecutive benchmark-positive sites in the "
            "same dsRNA region and arm"
        ),
        "total_rows": total_rows,
        "parsed_rows": parsed_rows,
        "regions_total": len({key for key, _ in regions}),
        "label_positive": positive_count,
        "label_negative": negative_count,
        "label_intermediate": intermediate_count,
        "negatives_with_positive_in_region_arm": len(distances),
        "negatives_without_positive_in_region_arm": negatives_without_positive,
        "defined_distance_min": min(distances) if distances else None,
        "defined_distance_median": (
            float(statistics.median(distances)) if distances else None
        ),
        "defined_distance_mean": (
            sum(distances) / len(distances) if distances else None
        ),
        "positive_spacing_min": (
            min(positive_spacings) if positive_spacings else None
        ),
        "positive_spacing_median": (
            float(statistics.median(positive_spacings))
            if positive_spacings
            else None
        ),
        "n_positive_spacings": len(positive_spacings),
        "threshold_rows": within_rows,
        "distance_bins": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--species",
        action="append",
        required=True,
        type=parse_species,
        metavar="NAME=CSV",
        help="Species name and full pre-balancing CSV; repeat per species.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--positive-threshold", type=float, default=0.15)
    parser.add_argument("--negative-threshold", type=float, default=0.001)
    args = parser.parse_args()

    results = [
        analyze(
            name,
            path,
            positive_threshold=args.positive_threshold,
            negative_threshold=args.negative_threshold,
        )
        for name, path in args.species
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
