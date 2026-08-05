#!/usr/bin/env python3
"""Validate training-dynamics summaries and manuscript figure provenance."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
EXPECTED_THRESHOLDS = {
    "baseline": {
        "Artery": 0.450,
        "Brain": 0.375,
        "Liver": 0.525,
        "MuscleSkeletal": 0.350,
        "Combined": 0.475,
        "Octopus": 0.600,
        "Ptychodera": 0.475,
        "Strongylocentrotus": 0.400,
    },
    "bioaware": {
        "Artery": 0.400,
        "Brain": 0.250,
        "Liver": 0.375,
        "MuscleSkeletal": 0.100,
        "Combined": 0.250,
        "Octopus": 0.325,
        "Ptychodera": 0.100,
        "Strongylocentrotus": 0.375,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    json_path = BASE / "training_dynamics_summary.json"
    csv_path = BASE / "training_dynamics_summary.csv"
    payload = json.loads(json_path.read_text())
    rows = list(csv.DictReader(csv_path.open()))
    if len(payload["runs"]) != 16 or len(rows) != 16:
        raise RuntimeError("Expected exactly 16 model histories")

    observed = {
        variant: {
            row["setting"]: float(row["selected_threshold"])
            for row in payload["runs"]
            if row["architecture"] == variant
        }
        for variant in EXPECTED_THRESHOLDS
    }
    for variant, expected_values in EXPECTED_THRESHOLDS.items():
        if set(observed[variant]) != set(expected_values):
            raise RuntimeError(f"{variant}: unexpected or missing settings")
        for setting, expected in expected_values.items():
            if abs(observed[variant][setting] - expected) > 1e-12:
                raise RuntimeError(
                    f"{variant}/{setting}: validation-selected threshold "
                    f"{observed[variant][setting]} differs from {expected}"
                )

    expected_summary = {
        "baseline": (0.4625, 0.350, 0.600),
        "bioaware": (0.2875, 0.100, 0.400),
    }
    for variant, (median, minimum, maximum) in expected_summary.items():
        summary = payload["threshold_summary"][variant]
        if (
            abs(float(summary["median"]) - median) > 1e-12
            or abs(float(summary["minimum"]) - minimum) > 1e-12
            or abs(float(summary["maximum"]) - maximum) > 1e-12
        ):
            raise RuntimeError(f"{variant}: threshold summary mismatch")

    analysis_png = BASE / "training_dynamics.png"
    analysis_pdf = BASE / "training_dynamics.pdf"
    manuscript_png = REPO / "manuscript" / "training_dynamics.png"
    for path in (analysis_png, analysis_pdf, manuscript_png):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
    if sha256(analysis_png) != sha256(manuscript_png):
        raise RuntimeError("Analysis and manuscript PNG copies differ")

    print("PASS: training-dynamics validation")
    print("  16/16 histories contain epochs 1--1000 and match their summaries")
    print("  baseline threshold median=0.4625, range=0.350--0.600")
    print("  bio-aware threshold median=0.2875, range=0.100--0.400")
    print(f"  figure sha256: {sha256(analysis_png)}")


if __name__ == "__main__":
    main()
