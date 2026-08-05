#!/usr/bin/env python3
"""Validate the inputs, claims, and outputs for Supplementary Fig. S2."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
TABLE = BASE / "data" / "gtex_v8_adar_expression.csv"

EXPECTED = {
    "Brain_Cerebellum": (107.38, 51.42),
    "Artery_Tibial": (94.35, 124.25),
    "Liver": (39.37, 2.59),
    "Muscle_Skeletal": (15.26, 5.63),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    table = pd.read_csv(TABLE)
    if len(table) != 54 or table["tissue"].nunique() != 54:
        raise RuntimeError("Expected one row for each of 54 GTEx v8 tissues")
    indexed = table.set_index("tissue")

    for tissue, expected in EXPECTED.items():
        observed = tuple(
            indexed.loc[
                tissue, ["ADAR_median_TPM", "ADARB1_median_TPM"]
            ].astype(float)
        )
        if any(abs(a - b) > 0.005 for a, b in zip(observed, expected)):
            raise RuntimeError(
                f"Unexpected GTEx values for {tissue}: {observed} != {expected}"
            )

    if indexed["ADARB1_median_TPM"].idxmax() != "Artery_Tibial":
        raise RuntimeError("Artery Tibial is not the maximum-ADARB1 tissue")
    if indexed["ADAR_median_TPM"].idxmin() != "Muscle_Skeletal":
        raise RuntimeError("Muscle Skeletal is not the minimum-ADAR tissue")

    outputs = [
        BASE / "figures" / "gtex_tissues.png",
        BASE / "figures" / "gtex_tissues.pdf",
        REPO / "manuscript" / "gtex_tissues.png",
    ]
    for path in outputs:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")

    if sha256(outputs[0]) != sha256(outputs[2]):
        raise RuntimeError("Analysis and manuscript PNG copies differ")

    print("PASS: Supplementary Fig. S2 output validation")
    print("  GTEx tissues: 54")
    print("  highest ADARB1: Artery_Tibial")
    print("  lowest ADAR: Muscle_Skeletal")
    print(f"  figure sha256: {sha256(outputs[0])}")


if __name__ == "__main__":
    main()
