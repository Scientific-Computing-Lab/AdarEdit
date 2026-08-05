#!/usr/bin/env python3
"""Validate the complete regenerated attention analysis against reference results."""

from __future__ import annotations

import json
from pathlib import Path


ANALYSIS = Path(__file__).resolve().parents[1]
DATA = ANALYSIS / "data"
FIGURES = ANALYSIS / "figures"


def compare_metrics(name: str, observed: dict, expected: dict, tolerance: float) -> None:
    for metric, reference in expected.items():
        difference = abs(float(observed[metric]) - float(reference))
        if difference > tolerance:
            raise RuntimeError(
                f"{name}/{metric}: expected {reference}, observed "
                f"{observed[metric]} (difference {difference})"
            )


def main() -> None:
    reference = json.loads((ANALYSIS / "reference_results.json").read_text())
    tolerance = float(reference["metric_absolute_tolerance"])

    environment = json.loads((DATA / "runtime_environment.json").read_text())
    if environment.get("status") != "PASS":
        raise RuntimeError("Dedicated environment verification did not pass")

    metrics = json.loads((DATA / "metrics_L2.json").read_text())
    if metrics["protocol"]["test_used_for_xgboost_fitting_or_feature_selection"]:
        raise RuntimeError("Test data were used for fitting or feature selection")
    if metrics["split_integrity"]["overlapping_duplexes"] != 0:
        raise RuntimeError("Validation/test duplex overlap detected")
    counts = reference["expected_counts"]
    integrity = metrics["split_integrity"]
    for field in (
        "validation_sites",
        "validation_duplexes",
        "test_sites",
        "test_duplexes",
    ):
        if int(integrity[field]) != int(counts[field]):
            raise RuntimeError(f"Unexpected {field}: {integrity[field]}")
    if metrics["top20_features_ranked"] != reference["primary_top20_features"]:
        raise RuntimeError("Primary validation-SHAP top-20 ranking changed")
    for model_name, expected in reference["primary_test_metrics"].items():
        compare_metrics(
            model_name,
            metrics[model_name]["test"],
            expected,
            tolerance,
        )

    test_metadata = json.loads((DATA / "attention_test_metadata.json").read_text())
    if int(test_metadata["node_level_test_rows"]) != int(counts["test_node_rows"]):
        raise RuntimeError("Unexpected node-level test row count")
    if float(test_metadata["maximum_probability_error_vs_shipped_test_predictions"]) > 1e-3:
        raise RuntimeError("GAT probabilities do not reproduce the checkpoint predictions")

    availability = json.loads((DATA / "availability_control_metrics.json").read_text())
    if availability.get("status") != "PASS":
        raise RuntimeError("Availability-control status is not PASS")
    if availability["split_integrity"]["overlapping_duplexes"] != 0:
        raise RuntimeError("Availability control has validation/test leakage")
    for split, field in (
        ("valid", "complete_window_validation_sites"),
        ("test", "complete_window_test_sites"),
    ):
        observed = availability["split_counts"][split]["complete_window_sites"]
        if int(observed) != int(counts[field]):
            raise RuntimeError(f"Unexpected {split} complete-window count")
    for model_name, expected in reference["availability_test_metrics"].items():
        compare_metrics(
            model_name,
            availability[model_name]["metrics"]["test"],
            expected,
            tolerance,
        )
    complete_ranking = availability["complete_window_attention"][
        "top20_positions_ranked_on_validation"
    ]
    if complete_ranking[:3] != reference["complete_window_leading_positions"]:
        raise RuntimeError("Complete-window leading SHAP positions changed")

    required_outputs = (
        DATA / "attention_valid_L2.csv",
        DATA / "attention_test_L2.csv",
        DATA / "node_level_attention_test_L2.csv",
        DATA / "predictions_test_L2.csv",
        DATA / "shap_validation_all_positions_L2.pkl",
        DATA / "shap_validation_top20_L2.pkl",
        FIGURES / "attention_interpretability.png",
        FIGURES / "attention_availability_control.png",
    )
    for path in required_outputs:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")

    print(
        "PASS: environment, split integrity, GAT fidelity, metrics, SHAP "
        "rankings, sensitivity controls, and figures validated"
    )


if __name__ == "__main__":
    main()
