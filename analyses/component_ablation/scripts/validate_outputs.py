#!/usr/bin/env python3
"""Validate component-ablation sources, metrics, and figures."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]

EXPECTED = {
    "full": (0.8543441226575810, 0.9100288409010573),
    "no_neighbors": (0.7909131010145567, 0.8560294164903983),
    "shuffle_pair_edges": (0.8256148903168624, 0.8858992189375510),
    "no_pair_edges": (0.8356726408982941, 0.8912414961164260),
    "untyped_edges": (0.8424989001319841, 0.9082163015596676),
    "no_seq_branch": (0.8483025354533734, 0.9045968937078349),
    "no_geometry": (0.8598335523434077, 0.9218277135398650),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    payload = json.loads((BASE / "ablation_summary.json").read_text())
    rows = {row["tag"]: row for row in payload["rows"]}
    if set(rows) != set(EXPECTED):
        raise RuntimeError("Unexpected or missing ablation rows")
    if payload["repeated_seeds"] is not False or payload["seed"] != 42:
        raise RuntimeError("Ablation seed provenance is incorrect")

    expected_hashes = {
        split: sha256(REPO / "data" / "human" / "Liver" / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }
    if payload["data_sha256"] != expected_hashes:
        raise RuntimeError("Ablation data hashes do not match canonical Liver")

    for tag, (expected_f1, expected_auc) in EXPECTED.items():
        row = rows[tag]
        if abs(float(row["f1"]) - expected_f1) > 1e-12:
            raise RuntimeError(f"{tag}: unexpected F1")
        if abs(float(row["auroc"]) - expected_auc) > 1e-12:
            raise RuntimeError(f"{tag}: unexpected AUROC")
        source = Path(row["source_summary"])
        if not source.is_absolute():
            source = REPO / source
        if not source.is_file() or sha256(source) != row["source_sha256"]:
            raise RuntimeError(f"{tag}: source-summary hash mismatch")

    csv_rows = list(
        csv.DictReader((BASE / "ablation_summary.csv").open())
    )
    if [row["tag"] for row in csv_rows] != [
        row["tag"] for row in payload["rows"]
    ]:
        raise RuntimeError("CSV and JSON row order differ")

    outputs = [
        BASE / "figures" / "component_ablation.png",
        BASE / "figures" / "component_ablation.pdf",
        REPO / "manuscript" / "figS_ablation.png",
    ]
    for output in outputs:
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {output}")
    if sha256(outputs[0]) != sha256(outputs[2]):
        raise RuntimeError("Analysis and manuscript PNG copies differ")

    print("PASS: Liver component-ablation validation")
    for tag in EXPECTED:
        row = rows[tag]
        print(
            f"  {tag:20s} F1={row['f1']:.4f} "
            f"AUROC={row['auroc']:.4f}"
        )
    print(f"  figure sha256: {sha256(outputs[0])}")


if __name__ == "__main__":
    main()
