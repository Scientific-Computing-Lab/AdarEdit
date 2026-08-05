#!/usr/bin/env python3
"""Check that every shipped input required by the ablation workflow exists."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ABLATIONS = (
    "no_neighbors",
    "shuffle_pair_edges",
    "no_pair_edges",
    "untyped_edges",
    "no_seq_branch",
    "no_geometry",
)


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Required input is missing or empty: {path}\n"
            "Restore the missing file at the repository-relative location "
            "documented in analyses/component_ablation/README.md."
        )


def main() -> None:
    required = [
        REPO / "environment.yml",
        REPO / "data" / "human" / "Liver" / "train.jsonl",
        REPO / "data" / "human" / "Liver" / "valid.jsonl",
        REPO / "data" / "human" / "Liver" / "test.jsonl",
        REPO / "checkpoints" / "bioaware_Liver" / "best.pth",
        REPO / "checkpoints" / "bioaware_Liver" / "summary.json",
    ]
    for tag in ABLATIONS:
        required.extend(
            [
                REPO
                / "checkpoints"
                / "ablations"
                / "Liver"
                / tag
                / "best.pth",
                REPO
                / "checkpoints"
                / "ablations"
                / "Liver"
                / tag
                / "summary.json",
            ]
        )
    for path in required:
        require_file(path)

    run_dirs = [REPO / "checkpoints" / "bioaware_Liver"]
    run_dirs.extend(
        REPO / "checkpoints" / "ablations" / "Liver" / tag
        for tag in ABLATIONS
    )
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        config = json.loads((run_dir / "run_config.json").read_text())
        if summary.get("graph_version") != "graph_v2":
            raise RuntimeError(f"{run_dir}: unrecognized graph version")
        if summary.get("selection_split") != "valid":
            raise RuntimeError(f"{run_dir}: model was not selected on validation")
        if summary.get("test_used_during_training", False):
            raise RuntimeError(f"{run_dir}: test was used during training")
        expected_corrections = {
            "baseline_pair_edge_multiplicity_per_direction": 1,
            "stem_feature": "true_contiguous_integer_stem_length",
            "sequential_edge_pair_compatibility": 0.0,
            "t_and_u_share_one_hot_channel": True,
        }
        for key, expected in expected_corrections.items():
            if config.get(key) != expected:
                raise RuntimeError(
                    f"{run_dir}: graph invariant {key} is "
                    f"{config.get(key)!r}, expected {expected!r}"
                )

    print("PASS: all shipped component-ablation inputs are present")
    print("PASS: all seven runs record the graph invariants")
    print(f"  repository: {REPO}")
    print("  data: data/human/Liver/{train,valid,test}.jsonl")
    print("  full model: checkpoints/bioaware_Liver")
    print("  ablations: checkpoints/ablations/Liver/<tag>")


if __name__ == "__main__":
    main()
