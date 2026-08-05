#!/usr/bin/env python3
"""Validate all inputs and outputs for Supplementary Figure S1."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS = Path(__file__).resolve().parents[1]
REPRO = Path(__file__).resolve().parents[3]
DATA = ANALYSIS / "data"
FIGURES = ANALYSIS / "figures"
CONTEXTS = ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]
VARIANTS = ["baseline", "bioaware"]
EXPECTED = {
    "Artery": 6638,
    "Brain": 6122,
    "Liver": 4893,
    "MuscleSkeletal": 4549,
    "Combined": 8286,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    provenance = json.loads((DATA / "cohort_provenance.json").read_text())
    if provenance.get("status") != "PASS":
        raise RuntimeError("Cohort provenance is not PASS")
    if provenance.get("total_test_intermediate_rows") != sum(EXPECTED.values()):
        raise RuntimeError("Unexpected total intermediate-site count")
    for context, expected in EXPECTED.items():
        observed = provenance["contexts"][context]["test_intermediate_rows"]
        if observed != expected:
            raise RuntimeError(f"{context}: expected {expected}, found {observed}")
        jsonl = DATA / "cohorts" / f"inter_{context}.jsonl"
        levels_path = DATA / "cohorts" / f"inter_{context}_levels.csv"
        levels = pd.read_csv(levels_path)
        jsonl_rows = sum(1 for _ in jsonl.open())
        if jsonl_rows != expected or len(levels) != expected:
            raise RuntimeError(f"{context}: cohort files do not contain {expected} rows")
        if not levels["editing_level_pct"].between(1, 15, inclusive="left").all():
            raise RuntimeError(f"{context}: cohort contains an invalid editing level")
        if provenance["contexts"][context]["jsonl_sha256"] != sha256(jsonl):
            raise RuntimeError(f"{context}: JSONL hash does not match provenance")
        if provenance["contexts"][context]["levels_sha256"] != sha256(levels_path):
            raise RuntimeError(f"{context}: level-table hash does not match provenance")

    scores = pd.read_csv(DATA / "intermediate_scores.csv")
    if len(scores) != 2 * sum(EXPECTED.values()):
        raise RuntimeError("Unexpected model-site score count")
    if not scores["editing_level"].between(1, 15, inclusive="left").all():
        raise RuntimeError("Intermediate scores contain an invalid editing level")
    if not scores["score"].between(0, 1, inclusive="both").all():
        raise RuntimeError("Predictions outside [0,1]")
    for context in CONTEXTS:
        for variant in VARIANTS:
            n = len(
                scores[
                    (scores["tissue"] == context)
                    & (scores["variant"] == variant)
                ]
            )
            if n != EXPECTED[context]:
                raise RuntimeError(f"{variant}/{context}: expected {EXPECTED[context]}, found {n}")
            prediction = DATA / "intermediate_predictions" / f"{variant}_{context}.csv"
            metadata_path = prediction.with_suffix(".metadata.json")
            metadata = json.loads(metadata_path.read_text())
            prediction_table = pd.read_csv(prediction)
            if metadata.get("status") != "PASS" or metadata.get("rows") != n:
                raise RuntimeError(f"{variant}/{context}: invalid prediction metadata")
            if metadata.get("output_sha256") != sha256(prediction):
                raise RuntimeError(f"{variant}/{context}: prediction hash mismatch")
            if not np.allclose(
                prediction_table["editing_level"],
                pd.read_csv(
                    DATA / "cohorts" / f"inter_{context}_levels.csv"
                )["editing_level_pct"],
                rtol=0,
                atol=1e-12,
            ):
                raise RuntimeError(f"{variant}/{context}: prediction rows are misaligned")

    score_table = pd.read_csv(DATA / "run1_score_by_bin.csv")
    threshold_table = pd.read_csv(DATA / "run2_threshold_auroc.csv")
    if len(score_table) != 10 or len(threshold_table) != 10:
        raise RuntimeError("Summary tables do not contain ten context/model rows")
    mean_columns = ["m_lt1", "m_1_5", "m_5_10", "m_10_15", "m_ge15"]
    for row in score_table.itertuples(index=False):
        values = [getattr(row, column) for column in mean_columns]
        if not all(left < right for left, right in zip(values, values[1:])):
            raise RuntimeError(f"Non-monotonic mean scores for {row.tissue}/{row.variant}")
    if not np.all(threshold_table["cut5"] <= threshold_table["cut10"]):
        raise RuntimeError("AUROC is not non-decreasing from 5% to 10%")
    if not np.all(threshold_table["cut10"] <= threshold_table["cut15"]):
        raise RuntimeError("AUROC is not non-decreasing from 10% to 15%")
    for row in threshold_table.itertuples(index=False):
        summary = json.loads(
            (
                REPRO
                / "checkpoints"
                / f"{row.variant}_{row.tissue}"
                / "summary.json"
            ).read_text()
        )
        canonical_auc = float(summary["test_metrics"]["auc"])
        if not np.isclose(row.cut15, canonical_auc, rtol=0, atol=1e-12):
            raise RuntimeError(
                f"{row.variant}/{row.tissue}: 15% AUROC differs from checkpoint"
            )

    analysis_metadata = json.loads((DATA / "analysis_metadata.json").read_text())
    if analysis_metadata.get("status") != "PASS":
        raise RuntimeError("Analysis metadata is not PASS")
    if analysis_metadata.get("intermediate_rows") != sum(EXPECTED.values()):
        raise RuntimeError("Analysis metadata has the wrong intermediate-site count")
    fidelity = analysis_metadata.get("canonical_inference_fidelity", [])
    if len(fidelity) != 10 or any(item.get("status") != "PASS" for item in fidelity):
        raise RuntimeError("Canonical checkpoint-inference fidelity is incomplete")

    for path in (
        FIGURES / "threshold_relaxation.png",
        FIGURES / "threshold_relaxation.pdf",
        FIGURES / "intermediate_site_scores.png",
        FIGURES / "intermediate_site_scores.pdf",
        REPRO / "manuscript" / "figS1_combined.png",
        REPRO / "manuscript" / "threshold_relaxation.png",
    ):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty figure: {path}")
    print(
        "PASS: 30,488 held-out intermediate records, 60,976 model-site "
        "scores, checkpoint fidelity, hashes, summary tables, and figures."
    )


if __name__ == "__main__":
    main()
