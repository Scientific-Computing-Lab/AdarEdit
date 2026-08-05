#!/usr/bin/env python3
"""Build long-format result tables from the canonical prediction archives."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


RESULTS_DIR = Path(__file__).resolve().parent
VARIANTS = ("baseline", "bioaware")
METRICS = ("f1", "precision", "recall", "auroc", "auprc", "threshold")


def load_matrix(variant: str, metric: str) -> dict[tuple[str, str], float]:
    path = RESULTS_DIR / f"matrix_{variant}_{metric}.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows or not reader.fieldnames:
        raise RuntimeError(f"{path}: empty matrix")
    row_key = reader.fieldnames[0]
    return {
        (row[row_key], context): float(row[context])
        for row in rows
        for context in reader.fieldnames[1:]
    }


def scalar(archive, key: str) -> str:
    return str(archive[key].item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="maximum allowed difference between recomputed and matrix metrics",
    )
    args = parser.parse_args()

    output_rows = []
    for variant in VARIANTS:
        matrices = {
            metric: load_matrix(variant, metric) for metric in METRICS
        }
        archives = sorted(
            (RESULTS_DIR / "preds").glob(f"{variant}__train-*__eval-*.npz")
        )
        if len(archives) != 64:
            raise RuntimeError(
                f"{variant}: expected 64 prediction archives, found {len(archives)}"
            )

        for path in archives:
            with np.load(path, allow_pickle=False) as archive:
                probabilities = np.asarray(archive["probs"], dtype=float)
                labels = np.asarray(archive["labels"], dtype=int)
                threshold = float(archive["threshold"])
                train_context = scalar(archive, "train_context")
                eval_context = scalar(archive, "eval_context")
                archive_variant = scalar(archive, "variant")
                graph_version = scalar(archive, "graph_version")

            if archive_variant != variant:
                raise RuntimeError(
                    f"{path}: variant {archive_variant!r} != {variant!r}"
                )
            predicted = probabilities >= threshold
            recomputed = {
                "f1": f1_score(labels, predicted),
                "precision": precision_score(labels, predicted, zero_division=0),
                "recall": recall_score(labels, predicted, zero_division=0),
                "auroc": roc_auc_score(labels, probabilities),
                "auprc": average_precision_score(labels, probabilities),
                "threshold": threshold,
            }
            key = (train_context, eval_context)
            for metric, value in recomputed.items():
                recorded = matrices[metric][key]
                if abs(value - recorded) > args.tolerance:
                    raise RuntimeError(
                        f"{path}: {metric}={value} does not match "
                        f"matrix value {recorded}"
                    )

            output_rows.append(
                {
                    "variant": variant,
                    "train_context": train_context,
                    "eval_context": eval_context,
                    "n": len(labels),
                    "positive_n": int(labels.sum()),
                    "threshold": recomputed["threshold"],
                    "f1": recomputed["f1"],
                    "precision": recomputed["precision"],
                    "recall": recomputed["recall"],
                    "auroc": recomputed["auroc"],
                    "auprc": recomputed["auprc"],
                    "graph_version": graph_version,
                    "prediction_file": str(path.relative_to(RESULTS_DIR)),
                }
            )

    table = pd.DataFrame(output_rows).sort_values(
        ["variant", "train_context", "eval_context"]
    )
    table.to_csv(RESULTS_DIR / "all_evaluations.csv", index=False)

    within_context = table[
        table["train_context"] == table["eval_context"]
    ].copy()
    within_context.to_csv(
        RESULTS_DIR / "table1_within_context.csv", index=False
    )
    print(
        f"PASS: wrote {len(table)} evaluations and "
        f"{len(within_context)} within-context rows"
    )


if __name__ == "__main__":
    main()
