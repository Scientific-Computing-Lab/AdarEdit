#!/usr/bin/env python3
"""Extract last-layer attention for one predefined Combined-data split.

The Baseline Combined GAT is frozen throughout this analysis.
This script creates one positional attention vector per candidate site. It
does not fit a model, select features, or use labels to construct features.

For each source position, attention is averaged across the four heads and
then max-aggregated over outgoing non-self edges. Positions -50 through +50
are retained for the downstream XGBoost probe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
sys.path.insert(0, str(ANALYSIS / "code"))

import gnnadar_verb_compact as baseline  # noqa: E402


BASE_ONE_HOT = {
    (1, 0, 0, 0, 0): "A",
    (0, 1, 0, 0, 0): "G",
    (0, 0, 1, 0, 0): "C",
    (0, 0, 0, 1, 0): "U",
    (0, 0, 0, 0, 1): "N",
}


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model_state(model, state):
    """Handle the homogeneous GAT projection-key rename across PyG releases."""
    expected = set(model.state_dict())
    compatible = dict(state)
    for key in list(state):
        if not key.endswith(".lin.weight") or key in expected:
            continue
        prefix = key[: -len("lin.weight")]
        source_key = prefix + "lin_src.weight"
        destination_key = prefix + "lin_dst.weight"
        if source_key in expected and destination_key in expected:
            weight = compatible.pop(key)
            compatible[source_key] = weight
            compatible[destination_key] = weight
    model.load_state_dict(compatible)


def assert_checkpoint_compatibility(checkpoint: dict):
    if checkpoint.get("variant") != "baseline":
        raise ValueError("The supplied checkpoint is not a Baseline model.")
    if checkpoint.get("context") != "Combined":
        raise ValueError("The supplied checkpoint is not the Combined model.")
    if checkpoint.get("selection_split") != "valid":
        raise ValueError("The checkpoint was not selected on validation data.")
    config = checkpoint.get("config", {})
    required = {
        "baseline_pair_edge_multiplicity_per_direction": 1,
        "t_and_u_share_one_hot_channel": True,
        "test_used_during_training": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(
                f"Checkpoint invariant {key!r} is {config.get(key)!r}; "
                f"expected {expected!r}."
            )


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch_load(checkpoint_path, device)
    assert_checkpoint_compatibility(checkpoint)
    model = baseline.RNAEditingGNN(
        input_dim=8,
        hidden_dim=32,
        output_dim=1,
        num_heads=4,
        num_layers=3,
        dropout_rate=0.2,
    ).to(device)
    load_model_state(model, checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def forward_with_last_attention(model, graph):
    """Run the model once and return its probability and layer-2 attention."""
    x = graph.x
    returned_attention = None
    for layer_index, (layer, batch_norm) in enumerate(
        zip(model.gat_layers, model.batch_norm_layers)
    ):
        if layer_index == len(model.gat_layers) - 1:
            x, returned_attention = layer(
                x,
                graph.edge_index,
                return_attention_weights=True,
            )
        else:
            x = layer(x, graph.edge_index)
        x = F.relu(x)
        x = batch_norm(x)
        x = model.dropout(x)
    pooled = global_mean_pool(x, graph.batch)
    probability = torch.sigmoid(model.fc(pooled)).reshape(-1)[0]
    return probability, returned_attention


def max_attention_by_source(
    returned_attention,
    node_features: np.ndarray,
    minimum_position: int,
    maximum_position: int,
):
    """Max-aggregate head-mean attention by source-node relative position."""
    returned_edges, coefficients = returned_attention
    edges = returned_edges.detach().cpu().numpy()
    values = coefficients.detach().cpu().numpy().mean(axis=1)
    mapping: dict[int, float] = {}
    for edge_offset in range(edges.shape[1]):
        source = int(edges[0, edge_offset])
        destination = int(edges[1, edge_offset])
        if source == destination:
            continue
        relative_position = int(round(float(node_features[source, 6])))
        if not minimum_position <= relative_position <= maximum_position:
            continue
        value = float(values[edge_offset])
        previous = mapping.get(relative_position)
        if previous is None or value > previous:
            mapping[relative_position] = value
    return mapping


def decode_base(node_feature_row: np.ndarray) -> str:
    key = tuple(int(round(value)) for value in node_feature_row[:5])
    return BASE_ONE_HOT.get(key, "N")


def load_threshold(summary_path: Path) -> float:
    summary = json.loads(summary_path.read_text())
    if summary.get("selection_split") != "valid":
        raise ValueError("summary.json does not record validation-only selection")
    threshold = float(summary["valid_selection_metrics"]["threshold"])
    if threshold != float(summary["test_metrics"]["threshold"]):
        raise ValueError("Test threshold differs from validation-selected threshold")
    return threshold


def duplex_group(sequence: str, structure: str) -> str:
    """Stable identifier used only to verify pair-disjoint split integrity."""
    material = (sequence + "\0" + structure).encode()
    return hashlib.sha256(material).hexdigest()[:24]


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    use_cuda = requested == "cuda" or (
        requested == "auto" and torch.cuda.is_available()
    )
    return torch.device("cuda" if use_cuda else "cpu")


def display_path(path: Path) -> str:
    """Display a repository-relative path without recording host locations."""
    try:
        return str(path.resolve().relative_to(ANALYSIS.resolve()))
    except ValueError:
        return path.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    parser.add_argument(
        "--input",
        type=Path,
        help="Override input JSONL; defaults to input/Combined_<split>.jsonl.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ANALYSIS / "checkpoint" / "best.pth",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ANALYSIS / "checkpoint" / "summary.json",
    )
    parser.add_argument(
        "--reference-test-predictions",
        type=Path,
        default=ANALYSIS / "checkpoint" / "test_predictions.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ANALYSIS / "data",
    )
    parser.add_argument("--feature-min", type=int, default=-50)
    parser.add_argument("--feature-max", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    input_path = args.input or ANALYSIS / "input" / f"Combined_{args.split}.jsonl"
    device = choose_device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)

    threshold = load_threshold(args.summary)
    model, checkpoint = load_model(args.checkpoint, device)
    expected_count = sum(1 for _ in input_path.open())
    reference = None
    if args.split == "test":
        reference = pd.read_csv(args.reference_test_predictions)
        if len(reference) != expected_count:
            raise ValueError(
                f"Test/reference row mismatch: {expected_count} versus "
                f"{len(reference)}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    positions = list(range(args.feature_min, args.feature_max + 1))
    output_path = args.out_dir / f"attention_{args.split}_L2.csv"
    node_output_path = (
        args.out_dir / "node_level_attention_test_L2.csv"
        if args.split == "test"
        else None
    )
    fields = (
        ["graph_index", "duplex_group"]
        + [f"pos_{position}" for position in positions]
        + ["model_probability", "model_prediction", "ground_truth"]
    )
    node_fields = [
        "graph_index",
        "duplex_group",
        "rel_pos",
        "base",
        "paired",
        "attention_L2",
        "ground_truth",
        "model_prediction",
        "correct",
    ]

    groups: set[str] = set()
    correct_count = 0
    graph_counts_by_label = {0: 0, 1: 0}
    correct_counts_by_label = {0: 0, 1: 0}
    node_row_count = 0
    maximum_probability_error = 0.0

    print(f"[attention:{args.split}] device={device}")
    print(f"[attention:{args.split}] checkpoint_epoch={checkpoint.get('epoch')}")
    print(f"[attention:{args.split}] threshold={threshold:.3f}")
    print(f"[attention:{args.split}] records={expected_count}")

    with ExitStack() as stack:
        input_handle = stack.enter_context(input_path.open())
        output_handle = stack.enter_context(output_path.open("w", newline=""))
        writer = csv.DictWriter(output_handle, fieldnames=fields)
        writer.writeheader()
        node_writer = None
        if node_output_path is not None:
            node_handle = stack.enter_context(node_output_path.open("w", newline=""))
            node_writer = csv.DictWriter(node_handle, fieldnames=node_fields)
            node_writer.writeheader()
        with torch.no_grad():
            for graph_index, line in enumerate(input_handle):
                record = json.loads(line)
                sequence, structure, target_index, label = baseline.parse_openai_json(
                    record
                )
                graph = baseline.create_rna_graph(
                    sequence,
                    structure,
                    target_index,
                    label,
                )
                if graph is None:
                    raise RuntimeError(f"Graph construction failed at row {graph_index}")

                group = duplex_group(sequence, structure)
                groups.add(group)
                graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
                graph = graph.to(device)
                probability_tensor, returned_attention = forward_with_last_attention(
                    model,
                    graph,
                )
                probability = float(probability_tensor.detach().cpu())
                prediction = int(probability >= threshold)
                integer_label = int(label)
                is_correct = int(prediction == integer_label)
                correct_count += is_correct
                graph_counts_by_label[integer_label] += 1
                correct_counts_by_label[integer_label] += is_correct

                if reference is not None:
                    reference_row = reference.iloc[graph_index]
                    if int(reference_row["label_from_loader"]) != int(label):
                        raise ValueError(
                            f"Reference label mismatch at test row {graph_index}"
                        )
                    maximum_probability_error = max(
                        maximum_probability_error,
                        abs(probability - float(reference_row["prob"])),
                    )

                node_features = graph.x.detach().cpu().numpy()
                attention_map = max_attention_by_source(
                    returned_attention,
                    node_features,
                    args.feature_min,
                    args.feature_max,
                )
                writer.writerow(
                    {
                        "graph_index": graph_index,
                        "duplex_group": group,
                        **{
                            f"pos_{position}": attention_map.get(position, 0.0)
                            for position in positions
                        },
                        "model_probability": f"{probability:.10g}",
                        "model_prediction": prediction,
                        "ground_truth": integer_label,
                    }
                )
                if node_writer is not None:
                    for node_index, node_feature_row in enumerate(node_features):
                        relative_position = int(round(float(node_feature_row[6])))
                        if not args.feature_min <= relative_position <= args.feature_max:
                            continue
                        node_writer.writerow(
                            {
                                "graph_index": graph_index,
                                "duplex_group": group,
                                "rel_pos": relative_position,
                                "base": decode_base(node_feature_row),
                                "paired": int(round(float(node_feature_row[5]))),
                                "attention_L2": f"{attention_map.get(relative_position, 0.0):.10g}",
                                "ground_truth": integer_label,
                                "model_prediction": prediction,
                                "correct": is_correct,
                            }
                        )
                        node_row_count += 1
                if graph_index % 250 == 0:
                    print(
                        f"[attention:{args.split}] "
                        f"processed={graph_index + 1}/{expected_count}",
                        flush=True,
                    )

    if reference is not None and maximum_probability_error > 1e-3:
        raise ValueError(
            "Test inference does not reproduce the shipped predictions: "
            f"max absolute error={maximum_probability_error:.6g}"
        )

    metadata = {
        "split": args.split,
        "graphs": expected_count,
        "duplexes": len(groups),
        "correct_predictions": correct_count,
        "graphs_by_label": {str(key): value for key, value in graph_counts_by_label.items()},
        "correct_predictions_by_label": {
            str(key): value for key, value in correct_counts_by_label.items()
        },
        "threshold": threshold,
        "threshold_source": "validation-selected checkpoint summary",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "attention_layer": 2,
        "attention_heads": 4,
        "feature_position_range": [args.feature_min, args.feature_max],
        "attention_aggregation": (
            "mean across heads, then maximum over outgoing non-self edges "
            "for each source position"
        ),
        "device": str(device),
        "maximum_probability_error_vs_shipped_test_predictions": (
            maximum_probability_error if reference is not None else None
        ),
        "node_level_test_table": (
            str(node_output_path.name) if node_output_path is not None else None
        ),
        "node_level_test_rows": node_row_count if node_output_path is not None else None,
    }
    metadata_path = args.out_dir / f"attention_{args.split}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[attention:{args.split}] output={display_path(output_path)}")
    if node_output_path is not None:
        print(
            f"[attention:{args.split}] "
            f"node_output={display_path(node_output_path)} "
            f"rows={node_row_count}"
        )
    print(
        f"[attention:{args.split}] metadata={display_path(metadata_path)}"
    )


if __name__ == "__main__":
    main()
