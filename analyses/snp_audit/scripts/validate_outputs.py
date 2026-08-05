#!/usr/bin/env python3
"""Validate the numerical inputs and outputs for Supplementary Fig. S4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(observed: float, expected: float, tolerance: float = 5e-7) -> None:
    if abs(observed - expected) > tolerance:
        raise RuntimeError(f"Unexpected value: {observed} != {expected}")


def main() -> None:
    burden = json.loads(
        (BASE / "data" / "duplex_snp_burden_summary.json").read_text()
    )
    donor = json.loads(
        (
            BASE
            / "donor_gtex_analysis"
            / "gtex_empirical_snp_summary.json"
        ).read_text()
    )
    overall = donor["overall_summary"]
    label_rows = {
        row["label_class"]: row for row in donor["label_summary"]
    }

    if int(burden["n_pairs"]) != 905:
        raise RuntimeError("Expected 905 reconstructed Alu-pair substrates")
    close(float(burden["snps_per_duplex_mean"]), 3.4828729281767954)
    close(float(burden["pct_duplex_with_ge1_common_snp"]), 92.48618784530387)

    total = int(overall["total_sites"])
    if total != 100377:
        raise RuntimeError(f"Expected 100,377 labeled sites, found {total}")
    close(100.0 * float(overall["any_coordinate_carrier_pct"]), 1.5142911224682946)
    close(
        100.0 * float(overall["transcript_equivalent_carrier_pct"]),
        0.15142911224682946,
    )
    close(
        100.0
        * int(overall["transcript_equivalent_carrier_sites_af_ge_0_01"])
        / total,
        0.040846028473654324,
    )
    close(
        100.0
        * int(overall["transcript_equivalent_carrier_sites_af_ge_0_05"])
        / total,
        0.029887326380950814,
    )
    close(
        100.0
        * float(
            label_rows["positive_any_tissue"][
                "transcript_equivalent_carrier_pct"
            ]
        ),
        0.17528877702368646,
    )
    close(
        100.0
        * float(
            label_rows["negative_only"][
                "transcript_equivalent_carrier_pct"
            ]
        ),
        0.14463444603727218,
    )

    outputs = [
        BASE / "figures" / "snp_audit.png",
        BASE / "figures" / "snp_audit.pdf",
        REPO / "manuscript" / "fig_snp.png",
    ]
    for path in outputs:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
    if sha256(outputs[0]) != sha256(outputs[2]):
        raise RuntimeError("Analysis and manuscript PNG copies differ")

    print("PASS: Supplementary Fig. S4 output validation")
    print(f"  reconstructed substrates: {int(burden['n_pairs']):,}")
    print(f"  labeled sites: {total:,}")
    print(
        "  strand-aware editing-mimicking overlap: "
        f"{100.0 * float(overall['transcript_equivalent_carrier_pct']):.3f}%"
    )
    print(f"  figure sha256: {sha256(outputs[0])}")


if __name__ == "__main__":
    main()
