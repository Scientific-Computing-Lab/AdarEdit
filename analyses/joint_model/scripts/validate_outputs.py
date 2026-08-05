#!/usr/bin/env python3
"""Recompute the published joint-model metrics from saved test predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RUNS = {
    "joint_baseline": ("baseline", "with_combined"),
    "joint_bioaware": ("bioaware", "with_combined"),
    "joint_baseline_noCombined": ("baseline", "no_combined"),
    "joint_bioaware_noCombined": ("bioaware", "no_combined"),
}
TOLERANCE = 1e-12


def close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=TOLERANCE):
        raise AssertionError(f"{label}: {actual} != {expected}")


def binary_metrics(labels, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    return {
        "n": int(len(labels)),
        "acc": float(accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(
            precision_score(labels, predicted, zero_division=0)
        ),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def read_predictions(path: Path):
    probabilities, labels, masks = [], [], []
    tissues = None
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            current = record["tissues"]
            if tissues is None:
                tissues = current
            elif current != tissues:
                raise AssertionError(f"Inconsistent tissue order in {path}")
            probabilities.append(record["prob"])
            labels.append(record["label"])
            masks.append(record["mask"])
    return (
        tissues,
        np.asarray(probabilities, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(masks, dtype=float),
    )


def validate_run(repository_root: Path, analysis_root: Path, run_name: str):
    variant, tissue_set = RUNS[run_name]
    source_dir = repository_root / "checkpoints" / run_name
    published_summary_path = source_dir / "summary.json"
    analysis_summary_path = analysis_root / "results" / run_name / "summary.json"
    summary = json.loads(published_summary_path.read_text())
    analysis_summary = json.loads(analysis_summary_path.read_text())
    if summary != analysis_summary:
        raise AssertionError(f"Stale analysis summary: {analysis_summary_path}")

    if summary["variant"] != variant or summary["tissue_set"] != tissue_set:
        raise AssertionError(f"Wrong run identity in {published_summary_path}")
    if summary["graph_version"] != "graph_v2":
        raise AssertionError(f"Wrong graph_version in {published_summary_path}")
    if summary["selection_split"] != "valid":
        raise AssertionError(f"Wrong selection split in {published_summary_path}")
    if summary["test_used_during_training"] is not False:
        raise AssertionError(f"Test-use flag is not false in {published_summary_path}")

    checkpoint_path = source_dir / "best.pth"
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["graph_version"] != "graph_v2":
        raise AssertionError(f"Wrong checkpoint graph version: {checkpoint_path}")
    if checkpoint["variant"] != variant:
        raise AssertionError(f"Wrong checkpoint variant: {checkpoint_path}")
    if checkpoint["tissues"] != summary["tissues"]:
        raise AssertionError(f"Wrong checkpoint tissue order: {checkpoint_path}")
    output_dimension = len(summary["tissues"])
    state = checkpoint["model_state"]
    output_weight = (
        state["fc.weight"] if variant == "baseline" else state["head.3.weight"]
    )
    if int(output_weight.shape[0]) != output_dimension:
        raise AssertionError(
            f"Output-head width mismatch in {checkpoint_path}: "
            f"{tuple(output_weight.shape)}"
        )

    tissues, probabilities, labels, masks = read_predictions(
        source_dir / "test_predictions.jsonl"
    )
    if tissues != summary["tissues"]:
        raise AssertionError(f"Tissue mismatch for {run_name}")
    if len(probabilities) != summary["sizes"]["test"]:
        raise AssertionError(f"Test-site count mismatch for {run_name}")

    recalculated = {}
    for index, tissue in enumerate(tissues):
        observed = masks[:, index] > 0
        recalculated[tissue] = binary_metrics(
            labels[observed, index],
            probabilities[observed, index],
            summary["thresholds_from_valid"][tissue],
        )
        expected = summary["test_by_tissue"][tissue]
        for metric, value in recalculated[tissue].items():
            if metric == "n":
                if value != expected[metric]:
                    raise AssertionError(
                        f"{run_name}/{tissue}/{metric}: {value} != {expected[metric]}"
                    )
            else:
                close(value, expected[metric], f"{run_name}/{tissue}/{metric}")

    for metric, summary_key in (
        ("f1", "test_macro_f1"),
        ("auc", "test_macro_auc"),
        ("auprc", "test_macro_auprc"),
    ):
        value = float(np.mean([recalculated[t][metric] for t in tissues]))
        close(value, summary[summary_key], f"{run_name}/{summary_key}")

    observed = masks > 0
    threshold_matrix = np.broadcast_to(
        np.asarray(
            [summary["thresholds_from_valid"][tissue] for tissue in tissues]
        ),
        probabilities.shape,
    )
    pooled_labels = labels[observed]
    pooled_probabilities = probabilities[observed]
    pooled_predictions = (
        pooled_probabilities >= threshold_matrix[observed]
    ).astype(int)
    pooled = {
        "n": int(len(pooled_labels)),
        "acc": float(accuracy_score(pooled_labels, pooled_predictions)),
        "f1": float(f1_score(pooled_labels, pooled_predictions, zero_division=0)),
        "precision": float(
            precision_score(pooled_labels, pooled_predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(pooled_labels, pooled_predictions, zero_division=0)
        ),
        "auc": float(roc_auc_score(pooled_labels, pooled_probabilities)),
        "auprc": float(
            average_precision_score(pooled_labels, pooled_probabilities)
        ),
    }
    for metric, value in pooled.items():
        expected = summary["test_pooled"][metric]
        if metric == "n":
            if value != expected:
                raise AssertionError(
                    f"{run_name}/pooled/{metric}: {value} != {expected}"
                )
        else:
            close(value, expected, f"{run_name}/pooled/{metric}")

    print(
        f"PASS {run_name}: sites={len(probabilities)} "
        f"observed_labels={int(masks.sum())} best_epoch={summary['best_epoch']}"
    )


def main() -> None:
    analysis_root = Path(__file__).resolve().parents[1]
    repository_root = Path(__file__).resolve().parents[3]
    for run_name in RUNS:
        validate_run(repository_root, analysis_root, run_name)
    print("PASS: all four joint-model outputs reproduce their summaries.")


if __name__ == "__main__":
    main()
