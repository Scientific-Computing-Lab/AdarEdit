#!/usr/bin/env python3
"""Validate the availability-control inputs, outputs and protocol invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ANALYSIS = Path(__file__).resolve().parents[1]
DATA = ANALYSIS / "data"
FIGURES = ANALYSIS / "figures"


def main() -> None:
    metrics = json.loads((DATA / "availability_control_metrics.json").read_text())
    if metrics.get("status") != "PASS":
        raise RuntimeError("analysis status is not PASS")
    if metrics["protocol"]["test_used_for_fitting_or_feature_selection"]:
        raise RuntimeError("test data were used for fitting or feature selection")
    if metrics["split_integrity"]["overlapping_duplexes"] != 0:
        raise RuntimeError("validation/test duplex overlap detected")

    expected = {
        "valid": (3793, 3150),
        "test": (4864, 4077),
    }
    for split, (sites, complete) in expected.items():
        observed = metrics["split_counts"][split]
        if (observed["sites"], observed["complete_window_sites"]) != (sites, complete):
            raise RuntimeError(f"unexpected {split} site counts: {observed}")
        for position in ("-2", "-1", "0", "1"):
            availability = metrics["proximal_position_availability"][split][position]
            if availability["available_n"] != availability["n"]:
                raise RuntimeError(f"{split}/pos_{position} is not universally available")

    table = pd.read_csv(DATA / "position_availability.csv")
    if len(table) != 2 * 2 * 101:
        raise RuntimeError("position-availability table has the wrong size")
    if not table["missing_fraction"].between(0, 1).all():
        raise RuntimeError("invalid missing fractions")

    required = (
        DATA / "availability_only_test_predictions.csv",
        DATA / "complete_window_test_predictions.csv",
        DATA / "shap_complete_window_all_positions.pkl",
        DATA / "shap_complete_window_top20.pkl",
        FIGURES / "attention_availability_control.png",
        FIGURES / "attention_availability_control.pdf",
    )
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty output: {path}")
    print("PASS: availability-control outputs and protocol invariants validated")


if __name__ == "__main__":
    main()
