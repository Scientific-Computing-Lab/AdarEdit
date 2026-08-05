#!/usr/bin/env python3
"""Summarize the Liver component-ablation runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
FULL = Path(
    os.environ.get(
        "ADAREDIT_ABLATION_FULL",
        REPO / "checkpoints" / "bioaware_Liver" / "summary.json",
    )
)
RUNS = Path(
    os.environ.get(
        "ADAREDIT_ABLATION_RUNS",
        REPO / "checkpoints" / "ablations" / "Liver",
    )
)

ABLATIONS = [
    ("no_neighbors", "Neighbouring-base context"),
    ("shuffle_pair_edges", "Correct pairing partner"),
    ("no_pair_edges", "Base-pair edges"),
    ("untyped_edges", "Edge typing"),
    ("no_seq_branch", "Sequence-CNN branch"),
    ("no_geometry", "Stem-loop geometry"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def load_summary(path: Path, expected_ablation: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text())
    if summary.get("variant") != "bioaware":
        raise RuntimeError(f"{path}: expected variant=bioaware")
    if summary.get("bio_ablation", "full") != expected_ablation:
        raise RuntimeError(
            f"{path}: expected bio_ablation={expected_ablation}"
        )
    if summary.get("selection_split") != "valid":
        raise RuntimeError(f"{path}: model was not selected on validation")
    if summary.get("test_used_during_training", False):
        raise RuntimeError(f"{path}: test was used during training")
    if int(summary.get("seed", 42)) != 42:
        raise RuntimeError(f"{path}: expected seed 42")
    if int(summary["epochs_requested"]) != 1000:
        raise RuntimeError(f"{path}: expected 1000 requested epochs")
    if summary["sizes"] != {"train": 12485, "valid": 3024, "test": 4150}:
        raise RuntimeError(f"{path}: unexpected split sizes")
    return summary


def result_row(
    tag: str,
    label: str,
    path: Path,
    summary: dict,
    full_metrics: dict,
) -> dict:
    metrics = summary["test_metrics"]
    return {
        "tag": tag,
        "component": label,
        "seed": int(summary.get("seed", 42)),
        "best_epoch": int(summary["best_epoch"]),
        "threshold": float(metrics["threshold"]),
        "f1": float(metrics["f1"]),
        "auroc": float(metrics["auc"]),
        "auprc": float(metrics["auprc"]),
        "delta_f1": float(metrics["f1"] - full_metrics["f1"]),
        "delta_auroc": float(metrics["auc"] - full_metrics["auc"]),
        "delta_auprc": float(metrics["auprc"] - full_metrics["auprc"]),
        "source_summary": display_path(path),
        "source_sha256": sha256(path),
    }


def main() -> None:
    full = load_summary(FULL, "full")
    full_metrics = full["test_metrics"]
    data_hashes = full["data_sha256"]
    rows = [
        result_row(
            "full",
            "Full bio-aware model",
            FULL,
            full,
            full_metrics,
        )
    ]
    for tag, label in ABLATIONS:
        path = RUNS / tag / "summary.json"
        summary = load_summary(path, tag)
        if summary["data_sha256"] != data_hashes:
            raise RuntimeError(f"{path}: data hashes differ from full model")
        rows.append(
            result_row(tag, label, path, summary, full_metrics)
        )

    payload = {
        "analysis": "single-component ablation of the bio-aware Liver model",
        "comparison": "ablation minus full bio-aware model",
        "selection_split": "valid",
        "test_used_during_training": False,
        "seed": 42,
        "repeated_seeds": False,
        "data_sha256": data_hashes,
        "rows": rows,
    }
    (BASE / "ablation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    columns = [
        "tag",
        "component",
        "seed",
        "best_epoch",
        "threshold",
        "f1",
        "auroc",
        "auprc",
        "delta_f1",
        "delta_auroc",
        "delta_auprc",
        "source_summary",
        "source_sha256",
    ]
    with (BASE / "ablation_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "Full bio-aware Liver: "
        f"F1={full_metrics['f1']:.4f} "
        f"AUROC={full_metrics['auc']:.4f} "
        f"AUPRC={full_metrics['auprc']:.4f}"
    )
    for row in sorted(rows[1:], key=lambda item: item["delta_auroc"]):
        print(
            f"{row['tag']:20s} "
            f"F1={row['f1']:.4f} ({row['delta_f1']:+.4f}) "
            f"AUROC={row['auroc']:.4f} ({row['delta_auroc']:+.4f})"
        )
    print(f"wrote {BASE / 'ablation_summary.json'}")
    print(f"wrote {BASE / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
