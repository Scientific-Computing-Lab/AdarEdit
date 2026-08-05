#!/usr/bin/env python3
"""Build complete held-out intermediate cohorts from full GTEx source CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
ANALYSIS = HERE.parents[1]
REPRO = HERE.parents[3]
DEFAULT_RAW = REPRO / "data" / "raw" / "editing_levels"
CONTEXTS = ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]
SOURCE_COLUMNS = [
    "L",
    "R",
    "structure",
    "Genomic_Location",
    "EditingIndex",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--repro", type=Path, default=REPRO)
    parser.add_argument("--output", type=Path, default=ANALYSIS / "data")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths(raw_dir: Path, context: str) -> list[Path]:
    base = raw_dir / f"{context}_Site_in_PairAlu_cov100.csv"
    if base.exists():
        return [base]
    parts = sorted(raw_dir.glob(f"{base.name}.part-*"))
    if not parts:
        raise FileNotFoundError(f"No source CSV or parts found for {context}: {base}")
    return parts


def read_source(paths: list[Path]) -> pd.DataFrame:
    if len(paths) == 1:
        return pd.read_csv(paths[0], usecols=SOURCE_COLUMNS, low_memory=False)
    # The .part files are byte chunks, not independent CSVs. Stream them as one
    # byte-identical file so quoted fields and rows crossing boundaries remain valid.
    process = subprocess.Popen(
        ["cat", *map(str, paths)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        table = pd.read_csv(process.stdout, usecols=SOURCE_COLUMNS, low_memory=False)
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Failed to concatenate source parts: {stderr}")
    return table


def parse_jsonl_sequence(line: str) -> str:
    record = json.loads(line)
    user = next(
        item["content"] for item in record["messages"] if item["role"] == "user"
    )
    fields = {}
    for field in user.split(", "):
        if ":" in field:
            key, value = field.split(":", 1)
            fields[key] = value
    return (fields.get("L", "") + fields.get("A", "") + fields.get("R", "")).upper().replace("T", "U")


def authoritative_split_map(repro: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for context in CONTEXTS:
        for split in ("train", "valid", "test"):
            path = repro / "data" / "human" / context / f"{split}.jsonl"
            with path.open() as handle:
                for line in handle:
                    sequence = parse_jsonl_sequence(line)
                    previous = mapping.setdefault(sequence, split)
                    if previous != split:
                        raise RuntimeError(
                            f"Cross-context split conflict for one substrate: "
                            f"{previous} versus {split}"
                        )
    if len(mapping) != 884:
        raise RuntimeError(f"Expected 884 substrates, found {len(mapping)}")
    return mapping


def editing_bin(levels: pd.Series) -> pd.Series:
    return pd.cut(
        levels,
        bins=[-np.inf, 1, 5, 10, 15, np.inf],
        labels=["<1", "1-5", "5-10", "10-15", ">=15"],
        right=False,
    )


def write_jsonl(table: pd.DataFrame, path: Path) -> None:
    with path.open("w") as handle:
        for row in table.itertuples(index=False):
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Predict if the central adenosine (A) in the given RNA "
                            "sequence context within an Alu element will be edited "
                            "to inosine (I) by ADAR enzymes."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"L:{row.L}, A:A, R:{row.R}, "
                            f"Alu Vienna Structure:{row.structure}"
                        ),
                    },
                    {"role": "assistant", "content": "no"},
                ]
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    cohort_dir = args.output / "cohorts"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    split_map = authoritative_split_map(args.repro)
    distribution_rows = []
    provenance = {
        "analysis": "Supplementary Figure S1 cohort construction",
        "authoritative_substrates": len(split_map),
        "contexts": {},
    }

    for context in CONTEXTS:
        paths = source_paths(args.raw_dir, context)
        raw = read_source(paths)
        for column in ("L", "R", "structure", "Genomic_Location"):
            raw[column] = raw[column].fillna("").astype(str)
        raw["EditingIndex"] = pd.to_numeric(raw["EditingIndex"], errors="coerce")
        levels = raw["EditingIndex"].dropna()
        counts = editing_bin(levels).value_counts(sort=False)
        n = int(counts.sum())
        distribution_rows.append(
            {
                "tissue": context,
                **{name: float(counts[name] / n) for name in counts.index},
                "intermediate": float(
                    (counts["1-5"] + counts["5-10"] + counts["10-15"]) / n
                ),
                "n": n,
            }
        )

        raw["sequence"] = (
            raw["L"] + "A" + raw["R"]
        ).str.upper().str.replace("T", "U", regex=False)
        raw["split"] = raw["sequence"].map(split_map)
        intermediate = raw[
            (raw["split"] == "test")
            & (raw["EditingIndex"] >= 1)
            & (raw["EditingIndex"] < 15)
        ].drop_duplicates(subset=["L", "R", "structure"], keep="first")
        intermediate = intermediate[
            intermediate["sequence"].str.len() == intermediate["structure"].str.len()
        ].reset_index(drop=True)

        jsonl = cohort_dir / f"inter_{context}.jsonl"
        levels_path = cohort_dir / f"inter_{context}_levels.csv"
        write_jsonl(intermediate, jsonl)
        pd.DataFrame(
            {
                "jsonl_line": np.arange(len(intermediate), dtype=int),
                "Genomic_Location": intermediate["Genomic_Location"],
                "editing_level_pct": intermediate["EditingIndex"],
            }
        ).to_csv(levels_path, index=False)

        if not intermediate["EditingIndex"].between(1, 15, inclusive="left").all():
            raise RuntimeError(f"{context}: invalid intermediate editing level")
        if not (intermediate["split"] == "test").all():
            raise RuntimeError(f"{context}: non-test substrate in intermediate cohort")

        provenance["contexts"][context] = {
            "source_parts": [
                {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
                for path in paths
            ],
            "source_rows": int(len(raw)),
            "source_rows_with_numeric_level": n,
            "test_intermediate_rows": int(len(intermediate)),
            "test_intermediate_substrates": int(intermediate["sequence"].nunique()),
            "jsonl": str(jsonl.relative_to(ANALYSIS)),
            "jsonl_sha256": sha256(jsonl),
            "levels": str(levels_path.relative_to(ANALYSIS)),
            "levels_sha256": sha256(levels_path),
            "status": "PASS",
        }
        print(
            f"[cohort] {context}: source={len(raw):,}, "
            f"test intermediate={len(intermediate):,}",
            flush=True,
        )

    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(args.output / "run0_distribution.csv", index=False)
    provenance["total_test_intermediate_rows"] = int(
        sum(
            item["test_intermediate_rows"]
            for item in provenance["contexts"].values()
        )
    )
    provenance["run0_distribution_sha256"] = sha256(
        args.output / "run0_distribution.csv"
    )
    provenance["status"] = "PASS"
    (args.output / "cohort_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"PASS: complete held-out intermediate cohort contains "
        f"{provenance['total_test_intermediate_rows']:,} rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
