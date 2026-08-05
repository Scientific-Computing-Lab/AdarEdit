#!/usr/bin/env python3
"""Run the Figure 6 in-silico mutagenesis analysis.

The analysis uses the homogeneous-edge Baseline Combined model and its
validation split. Panels B and D perturb existing RNAfold pair edges. Panel C
instead isolates sensitivity to the focal node's paired-state indicator: the
indicator is set to one or zero while sequence, edges, and every other feature
are held fixed. This feature-level comparison includes originally paired and
originally unpaired positions without inventing a pairing partner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import gnnadar_verb_compact as baseline  # noqa: E402


BASES = ("A", "G", "C", "T")
BASE_INDEX = {base: index for index, base in enumerate(BASES)}
CORE_POSITIONS = (-3, -2, -1, 1, 2, 3)
WIDE_POSITIONS = tuple(range(-40, 41))
INTERACTION_POSITIONS = (-1, 0, 1)
VALID_PAIRS = {
    ("A", "T"),
    ("T", "A"),
    ("G", "C"),
    ("C", "G"),
    ("G", "T"),
    ("T", "G"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(path: Path, device: torch.device):
    checkpoint = torch_load(path, device)
    model = baseline.RNAEditingGNN(
        input_dim=8,
        hidden_dim=32,
        output_dim=1,
        num_heads=4,
    ).to(device)
    state = checkpoint["model_state"]
    expected = model.state_dict()
    # PyG <2.3 named the homogeneous GAT projection ``lin_src``/``lin_dst``;
    # current PyG names the shared projection ``lin``. This compatibility
    # branch permits a local smoke test in an older environment. Published
    # outputs should use the environment pinned by the repository.
    if any(".lin_src.weight" in key for key in expected) and any(
        ".lin.weight" in key for key in state
    ):
        compatible = {}
        for key, value in state.items():
            if key.endswith(".lin.weight"):
                prefix = key[: -len("lin.weight")]
                compatible[prefix + "lin_src.weight"] = value
                compatible[prefix + "lin_dst.weight"] = value
            else:
                compatible[key] = value
        state = compatible
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def parse_pairs(structure: str) -> dict[int, int]:
    stack: list[int] = []
    partners: dict[int, int] = {}
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ValueError(f"Unbalanced dot-bracket string at index {index}")
            partner = stack.pop()
            partners[index] = partner
            partners[partner] = index
    if stack:
        raise ValueError("Unbalanced dot-bracket string: unmatched opening bracket")
    return partners


def load_records(path: Path):
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            sequence, structure, target_index, label = baseline.parse_openai_json(raw)
            if not sequence or not structure:
                raise ValueError(f"Could not parse input record at line {line_number}")
            graph = baseline.create_rna_graph(
                sequence, structure, target_index, label
            )
            if graph is None:
                raise ValueError(f"Could not build graph at line {line_number}")
            records.append(
                {
                    "sample_id": line_number - 1,
                    "line_number": line_number,
                    "sequence": sequence.replace("T", "U"),
                    "structure": structure,
                    "target_index": target_index,
                    "label": int(label),
                    "partners": parse_pairs(structure),
                    "graph": graph,
                }
            )
    return records


def validate_graphs(records) -> dict:
    duplicate_edge_graphs = 0
    flag_mismatches = 0
    target_non_a = 0
    for record in records:
        graph = record["graph"]
        edge_tuples = list(map(tuple, graph.edge_index.t().tolist()))
        if len(edge_tuples) != len(set(edge_tuples)):
            duplicate_edge_graphs += 1
        expected = {
            index for index in range(len(record["sequence"]))
            if index in record["partners"]
        }
        observed = {
            index for index, value in enumerate(graph.x[:, 5].tolist())
            if value == 1.0
        }
        if expected != observed:
            flag_mismatches += 1
        if record["sequence"][record["target_index"]] != "A":
            target_non_a += 1
    report = {
        "records": len(records),
        "duplicate_edge_graphs": duplicate_edge_graphs,
        "pair_flag_mismatch_graphs": flag_mismatches,
        "target_non_a_records": target_non_a,
    }
    if duplicate_edge_graphs or flag_mismatches or target_non_a:
        raise RuntimeError(f"Graph validation failed: {report}")
    return report


def model_probabilities(model, graphs, device, batch_size: int) -> np.ndarray:
    """Inference without the model's unused attention-weight return."""
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start : start + batch_size]).to(device)
            x = batch.x
            for index in range(model.num_layers):
                x = model.gat_layers[index](x, batch.edge_index)
                x = torch.relu(x)
                x = model.batch_norm_layers[index](x)
                x = model.dropout(x)
            pooled = global_mean_pool(x, batch.batch)
            probabilities = torch.sigmoid(model.fc(pooled)).view(-1)
            outputs.append(probabilities.cpu().numpy())
    if not outputs:
        return np.asarray([], dtype=float)
    return np.concatenate(outputs)


def clone_graph(graph, x=None, edge_index=None):
    return Data(
        x=graph.x.clone() if x is None else x,
        edge_index=graph.edge_index.clone() if edge_index is None else edge_index,
        y=graph.y.clone(),
    )


def mutate_base(graph, node_index: int, base: str):
    x = graph.x.clone()
    x[node_index, :5] = 0.0
    x[node_index, BASE_INDEX[base]] = 1.0
    return clone_graph(graph, x=x, edge_index=graph.edge_index.clone())


def disrupt_pair(graph, node_index: int, partner_index: int):
    x = graph.x.clone()
    x[node_index, 5] = 0.0
    x[partner_index, 5] = 0.0
    edges = graph.edge_index
    remove = (
        ((edges[0] == node_index) & (edges[1] == partner_index))
        | ((edges[0] == partner_index) & (edges[1] == node_index))
    )
    if int(remove.sum().item()) != 2:
        raise RuntimeError(
            "Expected exactly two directed edges for an existing base pair; "
            f"found {int(remove.sum().item())}"
        )
    return clone_graph(graph, x=x, edge_index=edges[:, ~remove].clone())


def set_pairing_indicator(graph, node_index: int, value: float):
    """Set only the focal node's paired-state feature and preserve topology."""
    if value not in (0.0, 1.0):
        raise ValueError(f"Pairing indicator must be 0 or 1, received {value}")
    x = graph.x.clone()
    x[node_index, 5] = value
    return clone_graph(graph, x=x, edge_index=graph.edge_index.clone())


def relative_node(record, position: int):
    node_index = record["target_index"] + position
    if node_index < 0 or node_index >= len(record["sequence"]):
        return None
    return node_index


def partner_at(record, position: int):
    node_index = relative_node(record, position)
    if node_index is None:
        return None, None
    return node_index, record["partners"].get(node_index)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, group_fields, value_fields):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        for field in value_fields:
            grouped[key][field].append(float(row[field]))
    output = []
    for key, values in sorted(grouped.items()):
        item = dict(zip(group_fields, key))
        n = len(next(iter(values.values())))
        item["n"] = n
        for field, observations in values.items():
            array = np.asarray(observations, dtype=float)
            item[f"mean_{field}"] = float(array.mean())
            item[f"sem_{field}"] = (
                float(array.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            )
        output.append(item)
    return output


def panel_a(model, selected, device, batch_size):
    rows = []
    for position in CORE_POSITIONS:
        eligible = [
            record for record in selected
            if relative_node(record, position) is not None
        ]
        for base in BASES:
            mutations = [
                mutate_base(record["graph"], relative_node(record, position), base)
                for record in eligible
            ]
            predictions = model_probabilities(
                model, mutations, device, batch_size
            )
            for record, prediction in zip(eligible, predictions):
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "position": position,
                        "base": base,
                        "prediction": float(prediction),
                    }
                )
    summary = summarize(rows, ["position", "base"], ["prediction"])
    position_means = defaultdict(list)
    for row in summary:
        position_means[int(row["position"])].append(row["mean_prediction"])
    centres = {
        position: float(np.mean(values))
        for position, values in position_means.items()
    }
    for row in summary:
        row["relative_preference"] = (
            row["mean_prediction"] - centres[int(row["position"])]
        )
    return rows, summary


def panel_b(model, selected, device, batch_size):
    rows = []
    for position in CORE_POSITIONS:
        eligible = []
        for record in selected:
            node_index, partner_index = partner_at(record, position)
            if partner_index is not None:
                eligible.append((record, node_index, partner_index))
        for base in BASES:
            retained_graphs = []
            disrupted_graphs = []
            for record, node_index, partner_index in eligible:
                retained = mutate_base(record["graph"], node_index, base)
                retained_graphs.append(retained)
                disrupted_graphs.append(
                    disrupt_pair(retained, node_index, partner_index)
                )
            retained_predictions = model_probabilities(
                model, retained_graphs, device, batch_size
            )
            disrupted_predictions = model_probabilities(
                model, disrupted_graphs, device, batch_size
            )
            for item, retained, disrupted in zip(
                eligible, retained_predictions, disrupted_predictions
            ):
                record = item[0]
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "position": position,
                        "base": base,
                        "retained_prediction": float(retained),
                        "disrupted_prediction": float(disrupted),
                        "delta": float(retained - disrupted),
                    }
                )
    summary = summarize(
        rows,
        ["position", "base"],
        ["retained_prediction", "disrupted_prediction", "delta"],
    )
    return rows, summary


def panel_c(model, selected, device, batch_size):
    rows = []
    for position in WIDE_POSITIONS:
        eligible = []
        paired_indicator_graphs = []
        unpaired_indicator_graphs = []
        for record in selected:
            node_index = relative_node(record, position)
            if node_index is None:
                continue
            eligible.append(record)
            paired_indicator_graphs.append(
                set_pairing_indicator(record["graph"], node_index, 1.0)
            )
            unpaired_indicator_graphs.append(
                set_pairing_indicator(record["graph"], node_index, 0.0)
            )
        paired_indicator_predictions = model_probabilities(
            model, paired_indicator_graphs, device, batch_size
        )
        unpaired_indicator_predictions = model_probabilities(
            model, unpaired_indicator_graphs, device, batch_size
        )
        for record, paired, unpaired in zip(
            eligible,
            paired_indicator_predictions,
            unpaired_indicator_predictions,
        ):
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "position": position,
                    "original_pairing_indicator": int(
                        record["graph"].x[
                            relative_node(record, position), 5
                        ].item()
                    ),
                    "paired_indicator_prediction": float(paired),
                    "unpaired_indicator_prediction": float(unpaired),
                    "delta": float(paired - unpaired),
                }
            )
    summary = summarize(
        rows,
        ["position"],
        [
            "paired_indicator_prediction",
            "unpaired_indicator_prediction",
            "delta",
        ],
    )
    return rows, summary


def panel_d(model, selected, device, batch_size):
    rows = []
    for position in INTERACTION_POSITIONS:
        eligible = []
        for record in selected:
            node_index, partner_index = partner_at(record, position)
            if partner_index is not None:
                eligible.append((record, node_index, partner_index))
        self_bases = ("A",) if position == 0 else BASES
        for self_base in self_bases:
            for partner_base in BASES:
                mutations = []
                pair_retained = (self_base, partner_base) in VALID_PAIRS
                for record, node_index, partner_index in eligible:
                    mutated = mutate_base(record["graph"], node_index, self_base)
                    mutated = mutate_base(mutated, partner_index, partner_base)
                    if not pair_retained:
                        mutated = disrupt_pair(
                            mutated, node_index, partner_index
                        )
                    mutations.append(mutated)
                predictions = model_probabilities(
                    model, mutations, device, batch_size
                )
                for item, prediction in zip(eligible, predictions):
                    record = item[0]
                    rows.append(
                        {
                            "sample_id": record["sample_id"],
                            "position": position,
                            "self_base": self_base,
                            "partner_base": partner_base,
                            "pair_retained": int(pair_retained),
                            "prediction": float(prediction),
                        }
                    )
    summary = summarize(
        rows,
        ["position", "self_base", "partner_base", "pair_retained"],
        ["prediction"],
    )
    return rows, summary


def validate_direct_inference(model, records, device):
    graph = records[0]["graph"]
    batch = Batch.from_data_list([graph]).to(device)
    with torch.inference_mode():
        canonical, _ = model(batch)
    direct = model_probabilities(model, [graph], device, 1)
    error = abs(float(canonical.view(-1)[0].item()) - float(direct[0]))
    if error > 1e-7:
        raise RuntimeError(f"Direct-inference validation failed: {error:.9g}")
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument(
        "--max-selected",
        type=int,
        default=None,
        help="Testing only: cap selected sites after deterministic file order.",
    )
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=("A", "B", "C", "D"),
        default=("A", "B", "C", "D"),
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    checkpoint_path = ROOT / "checkpoint" / "best.pth"
    input_path = ROOT / "input" / "Combined_valid.jsonl"
    output_dir = ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] root={ROOT}", flush=True)
    print(f"[setup] device={device}", flush=True)
    print(f"[setup] checkpoint={checkpoint_path}", flush=True)
    print(f"[setup] input={input_path}", flush=True)

    records = load_records(input_path)
    graph_validation = validate_graphs(records)
    model, checkpoint = load_model(checkpoint_path, device)
    direct_error = validate_direct_inference(model, records, device)

    original_predictions = model_probabilities(
        model, [record["graph"] for record in records], device, args.batch_size
    )
    selected = []
    for record, prediction in zip(records, original_predictions):
        record["original_probability"] = float(prediction)
        if record["label"] == 1 and prediction > args.confidence_threshold:
            selected.append(record)
    total_selected_before_cap = len(selected)
    if args.max_selected is not None:
        selected = selected[: args.max_selected]
    if not selected:
        raise RuntimeError("No validation records passed the selection criteria")

    selected_rows = [
        {
            "sample_id": record["sample_id"],
            "line_number": record["line_number"],
            "target_index": record["target_index"],
            "sequence_length": len(record["sequence"]),
            "original_probability": record["original_probability"],
        }
        for record in selected
    ]
    write_csv(
        output_dir / "selected_sites.csv",
        selected_rows,
        [
            "sample_id",
            "line_number",
            "target_index",
            "sequence_length",
            "original_probability",
        ],
    )

    outputs = {}
    if "A" in args.panels:
        print("[panel A] sequence substitutions", flush=True)
        rows, summary = panel_a(model, selected, device, args.batch_size)
        write_csv(
            output_dir / "panel_A_sequence_mutagenesis.csv",
            rows,
            ["sample_id", "position", "base", "prediction"],
        )
        write_csv(
            output_dir / "panel_A_summary.csv",
            summary,
            [
                "position",
                "base",
                "n",
                "mean_prediction",
                "sem_prediction",
                "relative_preference",
            ],
        )
        outputs["A"] = {"rows": len(rows), "summary_rows": len(summary)}

    if "B" in args.panels:
        print(
            "[panel B] retained versus disrupted existing pairs by base",
            flush=True,
        )
        rows, summary = panel_b(model, selected, device, args.batch_size)
        write_csv(
            output_dir / "panel_B_pair_disruption_by_base.csv",
            rows,
            [
                "sample_id",
                "position",
                "base",
                "retained_prediction",
                "disrupted_prediction",
                "delta",
            ],
        )
        write_csv(
            output_dir / "panel_B_summary.csv",
            summary,
            [
                "position",
                "base",
                "n",
                "mean_retained_prediction",
                "sem_retained_prediction",
                "mean_disrupted_prediction",
                "sem_disrupted_prediction",
                "mean_delta",
                "sem_delta",
            ],
        )
        outputs["B"] = {"rows": len(rows), "summary_rows": len(summary)}

    if "C" in args.panels:
        print(
            "[panel C] paired-state indicator sensitivity by position",
            flush=True,
        )
        rows, summary = panel_c(model, selected, device, args.batch_size)
        write_csv(
            output_dir / "panel_C_pairing_indicator_by_position.csv",
            rows,
            [
                "sample_id",
                "position",
                "original_pairing_indicator",
                "paired_indicator_prediction",
                "unpaired_indicator_prediction",
                "delta",
            ],
        )
        write_csv(
            output_dir / "panel_C_summary.csv",
            summary,
            [
                "position",
                "n",
                "mean_paired_indicator_prediction",
                "sem_paired_indicator_prediction",
                "mean_unpaired_indicator_prediction",
                "sem_unpaired_indicator_prediction",
                "mean_delta",
                "sem_delta",
            ],
        )
        outputs["C"] = {"rows": len(rows), "summary_rows": len(summary)}

    if "D" in args.panels:
        print("[panel D] focal/partner double substitutions", flush=True)
        rows, summary = panel_d(model, selected, device, args.batch_size)
        write_csv(
            output_dir / "panel_D_pair_interactions.csv",
            rows,
            [
                "sample_id",
                "position",
                "self_base",
                "partner_base",
                "pair_retained",
                "prediction",
            ],
        )
        write_csv(
            output_dir / "panel_D_summary.csv",
            summary,
            [
                "position",
                "self_base",
                "partner_base",
                "pair_retained",
                "n",
                "mean_prediction",
                "sem_prediction",
            ],
        )
        outputs["D"] = {"rows": len(rows), "summary_rows": len(summary)}

    metadata = {
        "status": "PASS",
        "analysis": "in_silico_mutagenesis",
        "model": "baseline_Combined",
        "architecture": "homogeneous-edge baseline GAT",
        "split": "validation",
        "selection": {
            "label": 1,
            "original_probability_operator": ">",
            "original_probability_threshold": args.confidence_threshold,
            "selected_before_optional_cap": total_selected_before_cap,
            "selected_analyzed": len(selected),
            "max_selected": args.max_selected,
        },
        "graph_validation": graph_validation,
        "direct_inference_max_abs_error": direct_error,
        "checkpoint": {
            "best_epoch": checkpoint.get("epoch"),
            "file_sha256": sha256(checkpoint_path),
        },
        "input": {
            "path": "input/Combined_valid.jsonl",
            "file_sha256": sha256(input_path),
            "records": len(records),
        },
        "device": str(device),
        "batch_size": args.batch_size,
        "panels": list(args.panels),
        "outputs": outputs,
        "panel_B_structural_intervention": (
            "retain an existing RNAfold pair versus remove both directed pair "
            "edges and clear the paired flags of both partners"
        ),
        "panel_C_feature_intervention": (
            "set only the focal node paired-state indicator to one versus zero; "
            "preserve sequence, all graph edges, and all other node features"
        ),
        "panel_D_structural_intervention": (
            "retain canonical or wobble focal-partner combinations; otherwise "
            "remove both directed pair edges and clear both paired flags"
        ),
        "refolding_performed": False,
    }
    with (output_dir / "analysis_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
