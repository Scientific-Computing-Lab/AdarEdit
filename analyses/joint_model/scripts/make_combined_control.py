#!/usr/bin/env python3
"""Build the with-Combined versus no-Combined joint-model control table."""

from __future__ import annotations

import csv
import json
from pathlib import Path

TISSUES = ["Artery", "Brain", "Liver", "MuscleSkeletal"]
VARIANTS = ["baseline", "bioaware"]
FIELDS = [
    "variant",
    "tissue",
    "n_test",
    "with_combined_f1",
    "no_combined_f1",
    "delta_f1_no_minus_with",
    "with_combined_auroc",
    "no_combined_auroc",
    "delta_auroc_no_minus_with",
]


def main() -> None:
    analysis_root = Path(__file__).resolve().parents[1]
    rows = []
    for variant in VARIANTS:
        with_combined = json.loads(
            (
                analysis_root
                / "results"
                / f"joint_{variant}"
                / "summary.json"
            ).read_text()
        )
        no_combined = json.loads(
            (
                analysis_root
                / "results"
                / f"joint_{variant}_noCombined"
                / "summary.json"
            ).read_text()
        )
        for tissue in TISSUES:
            with_metrics = with_combined["test_by_tissue"][tissue]
            no_metrics = no_combined["test_by_tissue"][tissue]
            if with_metrics["n"] != no_metrics["n"]:
                raise AssertionError(f"Test n differs for {variant}/{tissue}")
            rows.append(
                {
                    "variant": variant,
                    "tissue": tissue,
                    "n_test": with_metrics["n"],
                    "with_combined_f1": with_metrics["f1"],
                    "no_combined_f1": no_metrics["f1"],
                    "delta_f1_no_minus_with": (
                        no_metrics["f1"] - with_metrics["f1"]
                    ),
                    "with_combined_auroc": with_metrics["auc"],
                    "no_combined_auroc": no_metrics["auc"],
                    "delta_auroc_no_minus_with": (
                        no_metrics["auc"] - with_metrics["auc"]
                    ),
                }
            )

    output = analysis_root / "results/combined_supervision_control.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"PASS: wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
