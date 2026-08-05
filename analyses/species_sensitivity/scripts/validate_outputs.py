#!/usr/bin/env python3
"""Validate the numerical and graphical outputs for Supplementary Fig. S5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]

EXPECTED = {
    "Octopus": {
        "n": 1678,
        "n_pos": 929,
        "Baseline GAT": {
            "threshold": 0.600,
            "auroc": [0.8647, 0.8653, 0.8680, 0.8679],
            "f1": [0.8088, 0.8129, 0.8193, 0.8347],
        },
        "Bio-aware GNN": {
            "threshold": 0.325,
            "auroc": [0.8716, 0.8747, 0.8779, 0.8753],
            "f1": [0.8314, 0.8408, 0.8460, 0.8639],
        },
    },
    "Ptychodera": {
        "n": 1634,
        "n_pos": 875,
        "Baseline GAT": {
            "threshold": 0.475,
            "auroc": [0.8225, 0.8358, 0.8445, 0.8528],
            "f1": [0.7915, 0.8113, 0.8308, 0.8594],
        },
        "Bio-aware GNN": {
            "threshold": 0.100,
            "auroc": [0.8097, 0.8157, 0.8211, 0.8207],
            "f1": [0.7563, 0.7680, 0.7809, 0.8100],
        },
    },
    "S. purpuratus": {
        "n": 1493,
        "n_pos": 724,
        "Baseline GAT": {
            "threshold": 0.400,
            "auroc": [0.8601, 0.8700, 0.8777, 0.8885],
            "f1": [0.7968, 0.8134, 0.8281, 0.8591],
        },
        "Bio-aware GNN": {
            "threshold": 0.375,
            "auroc": [0.7968, 0.7997, 0.8014, 0.8105],
            "f1": [0.7337, 0.7445, 0.7575, 0.7962],
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    result = json.loads((BASE / "data" / "sensitivity.json").read_text())
    by_species = {block["short"]: block for block in result}
    if set(by_species) != set(EXPECTED):
        raise RuntimeError("Unexpected or missing species in sensitivity results")

    for species, expected in EXPECTED.items():
        block = by_species[species]
        for model in ("Baseline GAT", "Bio-aware GNN"):
            rows = block["results"][model]
            if len(rows) != 4:
                raise RuntimeError(f"{species}/{model}: expected four cohorts")
            if rows[0]["n"] != expected["n"] or rows[0]["n_pos"] != expected["n_pos"]:
                raise RuntimeError(f"{species}/{model}: unexpected full-test counts")
            if abs(rows[0]["threshold"] - expected[model]["threshold"]) > 1e-12:
                raise RuntimeError(f"{species}/{model}: unexpected threshold")
            for metric in ("auroc", "f1"):
                observed = [round(float(row[metric]), 4) for row in rows]
                if observed != expected[model][metric]:
                    raise RuntimeError(
                        f"{species}/{model}: unexpected {metric}: "
                        f"{observed} != {expected[model][metric]}"
                    )
            f1_values = [float(row["f1"]) for row in rows]
            if any(right < left for left, right in zip(f1_values, f1_values[1:])):
                raise RuntimeError(f"{species}/{model}: F1 is not monotonic")

            source = block["prediction_sources"][model]
            prediction = REPO / source["prediction_file"]
            summary = REPO / source["summary_file"]
            for path, key in (
                (prediction, "prediction_sha256"),
                (summary, "summary_sha256"),
            ):
                if not path.exists() or sha256(path) != source[key]:
                    raise RuntimeError(f"Source hash mismatch: {path}")

    raw = json.loads(
        (BASE / "raw_data" / "species_inter_site_distances.json").read_text()
    )
    expected_raw = {
        "Octopus bimaculoides": {
            "source_file": "Octopus_prebalancing.csv.gz",
            "source_sha256": "6a1e94248be006d3f33a6569042f9e0638d8f3aa2aa509938825f79d19c609aa",
            "negative": 30655,
            "positive": 16600,
            "without_positive": 4219,
            "within20": 0.46416571521774586,
            "spacing_min": 1,
            "spacing_median": 4.0,
            "n_spacings": 15095,
        },
        "Ptychodera flava": {
            "source_file": "Ptychodera_prebalancing.csv.gz",
            "source_sha256": "d6e9a8ec61a56959e7de96cfcf58d9c9aaef32fae82af5c2e0d6373c201863f6",
            "negative": 13083,
            "positive": 3839,
            "without_positive": 2150,
            "within20": 0.33486203470152104,
            "spacing_min": 1,
            "spacing_median": 4.0,
            "n_spacings": 3349,
        },
        "Strongylocentrotus purpuratus": {
            "source_file": "Strongylocentrotus_prebalancing.csv.gz",
            "source_sha256": "4cd73a168cd6cb3c8ac3c6bce5d61c67103befbd49d3f6b380ee28ffddad81fd",
            "negative": 27943,
            "positive": 6223,
            "without_positive": 5710,
            "within20": 0.2959596321082203,
            "spacing_min": 1,
            "spacing_median": 4.0,
            "n_spacings": 5314,
        },
    }
    for block in raw:
        expected = expected_raw[block["species"]]
        source = REPO / "data" / "species_prebalancing" / block["source_file"]
        if block["source_file"] != expected["source_file"]:
            raise RuntimeError(
                f"Unexpected panel-a source file for {block['species']}"
            )
        if not source.exists() or sha256(source) != expected["source_sha256"]:
            raise RuntimeError(f"Panel-a source hash mismatch: {source}")
        if block["source_sha256"] != expected["source_sha256"]:
            raise RuntimeError(
                f"Panel-a recorded source hash mismatch for {block['species']}"
            )
        if int(block["label_negative"]) != expected["negative"]:
            raise RuntimeError(
                f"Unexpected panel-a negative count for {block['species']}"
            )
        if int(block["label_positive"]) != expected["positive"]:
            raise RuntimeError(
                f"Unexpected panel-a positive count for {block['species']}"
            )
        if int(block["positive_spacing_min"]) != expected["spacing_min"]:
            raise RuntimeError(
                f"Unexpected positive-spacing minimum for {block['species']}"
            )
        if (
            float(block["positive_spacing_median"])
            != expected["spacing_median"]
        ):
            raise RuntimeError(
                f"Unexpected positive-spacing median for {block['species']}"
            )
        if int(block["n_positive_spacings"]) != expected["n_spacings"]:
            raise RuntimeError(
                f"Unexpected positive-spacing count for {block['species']}"
            )
        labelled_total = (
            int(block["label_negative"])
            + int(block["label_positive"])
            + int(block["label_intermediate"])
        )
        if labelled_total != int(block["parsed_rows"]):
            raise RuntimeError(
                f"Panel-a label classes do not cover all parsed rows for "
                f"{block['species']}"
            )
        if (
            int(block["negatives_without_positive_in_region_arm"])
            != expected["without_positive"]
        ):
            raise RuntimeError(
                f"Unexpected no-positive count for {block['species']}"
            )
        row20 = next(
            row for row in block["threshold_rows"] if row["threshold_bp"] == 20
        )
        if (
            abs(
                float(row20["frac_within_all_negatives"])
                - expected["within20"]
            )
            > 1e-12
        ):
            raise RuntimeError(
                f"Unexpected panel-a fraction for {block['species']}"
            )
        bin_total = sum(
            int(row["count"]) for row in block["distance_bins"]
        )
        if bin_total != expected["negative"]:
            raise RuntimeError(
                f"Panel-a bins do not cover every negative for {block['species']}"
            )

    outputs = [
        BASE / "figures" / "species_sensitivity.png",
        BASE / "figures" / "species_sensitivity.pdf",
        REPO / "manuscript" / "species_sensitivity.png",
    ]
    for path in outputs:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
    if sha256(outputs[0]) != sha256(outputs[2]):
        raise RuntimeError("Analysis and manuscript PNG copies differ")

    print("PASS: Supplementary Fig. S5 output validation")
    for species, expected in EXPECTED.items():
        values = []
        for model in ("Baseline GAT", "Bio-aware GNN"):
            rows = by_species[species]["results"][model]
            values.append(
                f"{model} F1={rows[0]['f1']:.3f}->{rows[-1]['f1']:.3f}"
            )
        print(f"  {species}: " + "; ".join(values))
    print(f"  figure sha256: {sha256(outputs[0])}")


if __name__ == "__main__":
    main()
