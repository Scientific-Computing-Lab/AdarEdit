#!/usr/bin/env python3
"""Validate the numerical and graphical outputs for Supplementary Fig. S6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
PERFORMANCE = BASE / "data" / "stability_performance.json"

EXPECTED_LOW_AUROC = {
    "Artery": {"Baseline GAT": 0.8963, "Bio-aware GNN": 0.8868},
    "Brain": {"Baseline GAT": 0.8829, "Bio-aware GNN": 0.8808},
    "Liver": {"Baseline GAT": 0.8984, "Bio-aware GNN": 0.8797},
    "MuscleSkeletal": {"Baseline GAT": 0.8389, "Bio-aware GNN": 0.8259},
    "Combined": {"Baseline GAT": 0.8995, "Bio-aware GNN": 0.8750},
}
EXPECTED_SITE_COUNTS = {
    "Artery": 4792,
    "Brain": 4566,
    "Liver": 4150,
    "MuscleSkeletal": 1425,
    "Combined": 4864,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    result = json.loads(PERFORMANCE.read_text())
    global_result = result["_global"]
    expected_global = {
        "n_substrates": 884,
        "q1": 0.7762,
        "median": 0.8120,
        "q3": 0.8442,
    }
    for key, expected in expected_global.items():
        observed = global_result[key]
        if abs(float(observed) - expected) > 5e-5:
            raise RuntimeError(f"Unexpected global {key}: {observed} != {expected}")

    for context, variants in EXPECTED_LOW_AUROC.items():
        if int(result[context]["n_sites"]) != EXPECTED_SITE_COUNTS[context]:
            raise RuntimeError(f"Unexpected aligned-site count for {context}")
        for variant, expected in variants.items():
            block = result[context]["performance"][variant]
            observed = float(block["low_stability"]["auroc"])
            if abs(observed - expected) > 5e-5:
                raise RuntimeError(
                    f"Unexpected low-stability AUROC for {context}/{variant}: "
                    f"{observed} != {expected}"
                )
            prediction = REPO / block["prediction_file"]
            if not prediction.exists():
                raise RuntimeError(f"Missing prediction source: {prediction}")
            if sha256(prediction) != block["prediction_sha256"]:
                raise RuntimeError(f"Prediction-source hash mismatch: {prediction}")

    outputs = [
        BASE / "figures" / "substrate_stability.png",
        BASE / "figures" / "substrate_stability.pdf",
        REPO / "manuscript" / "fig_stability.png",
    ]
    for path in outputs:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
    if sha256(outputs[0]) != sha256(outputs[2]):
        raise RuntimeError("Analysis and manuscript PNG copies differ")

    print("PASS: Supplementary Fig. S6 output validation")
    print(
        "  global paired-fraction Q1/median/Q3: "
        f"{global_result['q1']:.4f}/"
        f"{global_result['median']:.4f}/"
        f"{global_result['q3']:.4f}"
    )
    for context, variants in EXPECTED_LOW_AUROC.items():
        values = ", ".join(
            f"{variant}={result[context]['performance'][variant]['low_stability']['auroc']:.4f}"
            for variant in variants
        )
        print(f"  {context} low-stability AUROC: {values}")
    print(f"  figure sha256: {sha256(outputs[0])}")


if __name__ == "__main__":
    main()
