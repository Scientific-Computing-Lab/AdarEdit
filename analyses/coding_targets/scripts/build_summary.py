#!/usr/bin/env python3
"""Build the coding-target AUROC table from per-adenosine predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent

VARIANTS = ("baseline", "bioaware", "joint_baseline", "joint")
GENES = ("AJUBA", "BLCAP", "FLNA", "TTYH2", "NEIL1", "GRIA2", "GRIA3")
TISSUES = ("Brain", "Combined", "Liver")
FILE_TISSUE = {
    "Brain": "BrainCerebellum",
    "Combined": "Combined",
    "Liver": "Liver",
}
EDITED_MIN = 15.0
NON_EDITED_MAX = 1.0
MIN_PER_CLASS = 2


def prediction_path(root: Path, variant: str, gene: str, tissue: str) -> Path:
    return (
        root
        / variant
        / gene
        / f"adenosine_prediction_{FILE_TISSUE[tissue]}.csv"
    )


def summarize_file(path: Path):
    labels = []
    scores = []
    excluded = 0
    unmeasured = 0
    with path.open() as handle:
        for row in csv.DictReader(handle):
            raw_level = row["editing_level"].strip()
            if not raw_level:
                unmeasured += 1
                continue
            level = float(raw_level)
            if level >= EDITED_MIN:
                labels.append(1)
            elif level < NON_EDITED_MAX:
                labels.append(0)
            else:
                excluded += 1
                continue
            scores.append(float(row["prediction"]))
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    number_edited = int(labels_array.sum())
    number_not_edited = int(len(labels_array) - number_edited)
    auroc = None
    if (
        number_edited >= MIN_PER_CLASS
        and number_not_edited >= MIN_PER_CLASS
    ):
        auroc = float(roc_auc_score(labels_array, scores_array))
    return {
        "auroc": auroc,
        "n_edited": number_edited,
        "n_not_edited": number_not_edited,
        "n_intermediate_excluded": excluded,
        "n_unmeasured_excluded": unmeasured,
        "n_total_adenosines": (
            number_edited + number_not_edited + excluded + unmeasured
        ),
    }


def build(predictions_root: Path):
    rows = []
    for variant in VARIANTS:
        for tissue in TISSUES:
            for gene in GENES:
                path = prediction_path(predictions_root, variant, gene, tissue)
                if not path.is_file():
                    raise FileNotFoundError(path)
                rows.append(
                    {
                        "variant": variant,
                        "gene": gene,
                        "tissue": tissue,
                        **summarize_file(path),
                    }
                )
    return rows


def write_outputs(rows, csv_path: Path, json_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "variant",
        "gene",
        "tissue",
        "auroc",
        "n_edited",
        "n_not_edited",
        "n_intermediate_excluded",
        "n_unmeasured_excluded",
        "n_total_adenosines",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["auroc"] = (
                "" if row["auroc"] is None else f"{row['auroc']:.10g}"
            )
            writer.writerow(output)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-root", type=Path, default=PACKAGE / "predictions"
    )
    parser.add_argument(
        "--csv", type=Path, default=PACKAGE / "results" / "auroc_summary.csv"
    )
    parser.add_argument(
        "--json", type=Path, default=PACKAGE / "results" / "auroc_summary.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build(args.predictions_root)
    write_outputs(rows, args.csv, args.json)
    available = sum(row["auroc"] is not None for row in rows)
    print(
        f"[done] {len(rows)} model/gene/tissue cells; "
        f"{available} have at least {MIN_PER_CLASS} sites per class"
    )
    print(f"[done] {args.csv}")


if __name__ == "__main__":
    main()
