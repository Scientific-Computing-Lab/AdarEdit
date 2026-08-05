#!/usr/bin/env python3
"""Build the joint-versus-per-tissue table from supplied model outputs."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

TISSUES = ["Artery", "Brain", "Combined", "Liver", "MuscleSkeletal"]
VARIANTS = [
    ("baseline", "Baseline"),
    ("bioaware", "Bio-aware"),
]
FIELDS = ["tissue", "model", "f1", "auroc"]


def per_tissue_metrics(repository_root: Path, variant: str, tissue: str):
    prediction_path = (
        repository_root
        / "results/preds"
        / f"{variant}__train-{tissue}__eval-{tissue}.npz"
    )
    with np.load(prediction_path, allow_pickle=False) as archive:
        graph_version = str(archive["graph_version"].item())
        if graph_version != "graph_v2":
            raise AssertionError(
                f"{prediction_path} uses graph_version={graph_version}"
            )
        probabilities = archive["probs"].astype(float)
        labels = archive["labels"].astype(int)
        threshold = float(archive["threshold"].item())
    summary = json.loads(
        (
            repository_root
            / "checkpoints"
            / f"{variant}_{tissue}"
            / "summary.json"
        ).read_text()
    )
    if len(labels) != summary["test_metrics"]["n"]:
        raise AssertionError(f"Test n mismatch for {variant}/{tissue}")
    if not np.isclose(
        threshold,
        summary["test_metrics"]["threshold"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(f"Threshold mismatch for {variant}/{tissue}")
    predicted = (probabilities >= threshold).astype(int)
    return {
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probabilities)),
    }


def main() -> None:
    analysis_root = Path(__file__).resolve().parents[1]
    repository_root = Path(__file__).resolve().parents[3]
    rows = []

    joint_summaries = {}
    for variant, _ in VARIANTS:
        source = (
            repository_root
            / "checkpoints"
            / f"joint_{variant}"
            / "summary.json"
        )
        summary = json.loads(source.read_text())
        if summary["graph_version"] != "graph_v2":
            raise AssertionError(f"{source} is not graph_v2")
        if summary["selection_split"] != "valid":
            raise AssertionError(f"{source} was not selected on validation")
        if summary["test_used_during_training"] is not False:
            raise AssertionError(f"{source} reports test use during training")
        joint_summaries[variant] = summary
        destination = (
            analysis_root
            / "model_summaries"
            / f"joint_{variant}_summary.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    for tissue in TISSUES:
        for variant, label in VARIANTS:
            per_tissue = per_tissue_metrics(
                repository_root, variant, tissue
            )
            joint = joint_summaries[variant]["test_by_tissue"][tissue]
            rows.extend(
                [
                    {
                        "tissue": tissue,
                        "model": f"Per-tissue {label}",
                        **per_tissue,
                    },
                    {
                        "tissue": tissue,
                        "model": f"Joint {label}",
                        "f1": joint["f1"],
                        "auroc": joint["auc"],
                    },
                ]
            )

    output = analysis_root / "data/comparison_data.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"PASS: wrote {len(rows)} source-derived rows to {output}")


if __name__ == "__main__":
    main()
