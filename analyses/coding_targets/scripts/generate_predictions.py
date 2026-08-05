#!/usr/bin/env python3
"""Run the eight trained human models on seven coding-region RNA targets.

The graph builders and model classes are imported from ``code/``.  Those files
are exact copies of the implementations used to train the supplied model
checkpoints.  No training, fine-tuning, threshold selection, or coding-target
label information is used during inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
CODE = PACKAGE / "code"
sys.path.insert(0, str(CODE))

import bioaware_gnn as bio  # noqa: E402
import gnnadar_verb_compact as baseline  # noqa: E402
import train_joint as joint  # noqa: E402


GENES = ("AJUBA", "BLCAP", "FLNA", "TTYH2", "NEIL1", "GRIA2", "GRIA3")
TISSUES = ("Brain", "Combined", "Liver")
TABLE_TISSUE = {
    "Brain": "Brain_Cerebellum",
    "Combined": "Combined",
    "Liver": "Liver",
}
FILE_TISSUE = {
    "Brain": "BrainCerebellum",
    "Combined": "Combined",
    "Liver": "Liver",
}
JOINT_TISSUES = ("Artery", "Brain", "Combined", "Liver", "MuscleSkeletal")


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    sequence: str
    structure: str


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def read_records(path: Path) -> list[SequenceRecord]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) % 3:
        raise ValueError(f"{path}: expected three lines per FASTA/Vienna record")
    records = []
    for offset in range(0, len(lines), 3):
        header, raw_sequence, structure = lines[offset : offset + 3]
        if not header.startswith(">"):
            raise ValueError(f"{path}:{offset + 1}: expected a FASTA header")
        # This is the same normalization applied during model training.
        sequence = raw_sequence.upper().replace("T", "U")
        if len(sequence) != len(structure):
            raise ValueError(
                f"{path}: sequence/structure length mismatch "
                f"({len(sequence)} versus {len(structure)})"
            )
        records.append(SequenceRecord(header[1:] or "sequence", sequence, structure))
    return records


def sequence_path(data_root: Path, gene: str) -> Path:
    name = "seq_TTYH2.txt" if gene == "TTYH2" else "seq.txt"
    return data_root / gene / name


def table_path(data_root: Path, gene: str, tissue: str) -> Path:
    return (
        data_root
        / gene
        / f"dsRNA_structure_with_{TABLE_TISSUE[tissue]}_editing_sites_andA20.csv"
    )


def load_editing_levels(path: Path) -> dict[int, float]:
    frame = pd.read_csv(path)
    required = {"Local_Position", "EditingLevel"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    positions = pd.to_numeric(frame["Local_Position"], errors="raise").astype(int)
    levels = pd.to_numeric(frame["EditingLevel"], errors="coerce")
    if levels.isna().any():
        raise ValueError(f"{path}: EditingLevel contains missing/non-numeric values")
    if positions.duplicated().any():
        raise ValueError(f"{path}: Local_Position is not unique")
    return dict(zip(positions, levels.astype(float)))


def build_graphs(record: SequenceRecord, variant: str):
    graphs = []
    metadata = []
    for index, nucleotide in enumerate(record.sequence):
        if nucleotide != "A":
            continue
        if variant == "baseline":
            graph = baseline.create_rna_graph(
                record.sequence, record.structure, index, label=0
            )
        else:
            graph = bio.build_graph(
                bio.GraphExample(record.sequence, record.structure, index, 0),
                use_neighbors=True,
                use_geometry=True,
                plfold_dir=None,
            )
        if graph is None:
            raise RuntimeError(
                f"failed to build {variant} graph for "
                f"{record.sequence_id} position {index + 1}"
            )
        graphs.append(graph)
        metadata.append((index, index + 1))
    return graphs, metadata


def infer(model, graphs, device, model_kind: str, tissue_columns=None, batch_size=128):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    output = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if model_kind == "single_baseline":
                probabilities, _ = model(batch)
            elif model_kind == "single_bioaware":
                probabilities = torch.sigmoid(model(batch))
            else:
                probabilities = torch.sigmoid(model(batch))
                probabilities = probabilities[:, tissue_columns]
            output.append(probabilities.detach().cpu().numpy())
    return np.concatenate(output, axis=0)


def checkpoint_state(path: Path, device: torch.device):
    checkpoint = torch_load(path, device)
    state = checkpoint.get("model_state")
    if state is None:
        raise KeyError(f"{path}: checkpoint has no model_state")
    return checkpoint, state


def load_model_state(model, state):
    """Load checkpoints across the PyG GATConv key rename.

    PyG 2.6 stores a homogeneous GAT projection as ``lin.weight``. Older PyG
    releases expose the same shared projection through ``lin_src.weight`` and
    ``lin_dst.weight``. Duplicating that one trained matrix into the two aliases
    preserves the homogeneous GAT computation; no parameter is estimated or
    changed.
    """
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


def assert_checkpoint(checkpoint: dict, expected_variant: str, joint_model: bool):
    if checkpoint.get("variant") != expected_variant:
        raise ValueError(
            f"checkpoint variant {checkpoint.get('variant')!r} does not match "
            f"{expected_variant!r}"
        )
    config = checkpoint.get("config", {})
    required = {
        "baseline_pair_edge_multiplicity_per_direction": 1,
        "stem_feature": "true_contiguous_integer_stem_length",
        "sequential_edge_pair_compatibility": 0.0,
        "t_and_u_share_one_hot_channel": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(
                f"checkpoint invariant {key!r} is {config.get(key)!r}, "
                f"expected {expected!r}"
            )
    if joint_model and tuple(checkpoint.get("tissues", ())) != JOINT_TISSUES:
        raise ValueError("joint checkpoint tissue order is not the expected five-tissue order")


def write_predictions(
    path: Path,
    record: SequenceRecord,
    metadata,
    probabilities,
    editing_levels: dict[int, float],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "sequence_id",
        "adenosine_index",
        "local_position",
        "prediction",
        "editing_level",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (index, local_position), probability in zip(metadata, probabilities):
            writer.writerow(
                {
                    "sequence_id": record.sequence_id,
                    "adenosine_index": index,
                    "local_position": local_position,
                    "prediction": f"{float(probability):.10g}",
                    "editing_level": (
                        f"{editing_levels[local_position]:.10g}"
                        if local_position in editing_levels
                        else ""
                    ),
                }
            )


def output_path(out_root: Path, variant: str, gene: str, tissue: str) -> Path:
    return (
        out_root
        / variant
        / gene
        / f"adenosine_prediction_{FILE_TISSUE[tissue]}.csv"
    )


def run_single_models(
    data_root: Path,
    checkpoint_root: Path,
    out_root: Path,
    device: torch.device,
    batch_size: int,
):
    for variant in ("baseline", "bioaware"):
        graph_variant = variant
        for tissue in TISSUES:
            checkpoint_path = checkpoint_root / f"{variant}_{tissue}" / "best.pth"
            checkpoint, state = checkpoint_state(checkpoint_path, device)
            assert_checkpoint(checkpoint, variant, joint_model=False)

            first_record = read_records(sequence_path(data_root, GENES[0]))[0]
            first_graphs, _ = build_graphs(first_record, graph_variant)
            input_dimension = int(first_graphs[0].x.shape[1])
            if variant == "baseline":
                model = baseline.RNAEditingGNN(
                    input_dim=input_dimension,
                    hidden_dim=32,
                    output_dim=1,
                    num_heads=4,
                    num_layers=3,
                    dropout_rate=0.2,
                ).to(device)
                model_kind = "single_baseline"
            else:
                model = bio.BioAwareGNN(
                    in_dim=input_dimension,
                    hidden=96,
                    heads=4,
                    layers=3,
                    edge_emb_dim=6,
                    edge_scalar_dim=12,
                    dropout=0.1,
                    seq_branch_dim=128,
                    use_global_attn=False,
                    global_attn_heads=4,
                ).to(device)
                model_kind = "single_bioaware"
            load_model_state(model, state)

            print(f"[inference] {variant}/{tissue}: {checkpoint_path}")
            for gene in GENES:
                record = read_records(sequence_path(data_root, gene))[0]
                graphs, metadata = build_graphs(record, graph_variant)
                probabilities = infer(
                    model,
                    graphs,
                    device,
                    model_kind=model_kind,
                    batch_size=batch_size,
                ).reshape(-1)
                levels = load_editing_levels(table_path(data_root, gene, tissue))
                write_predictions(
                    output_path(out_root, variant, gene, tissue),
                    record,
                    metadata,
                    probabilities,
                    levels,
                )


def run_joint_models(
    data_root: Path,
    checkpoint_root: Path,
    out_root: Path,
    device: torch.device,
    batch_size: int,
):
    selected_columns = [JOINT_TISSUES.index(tissue) for tissue in TISSUES]
    for output_variant, checkpoint_name, model_variant in (
        ("joint_baseline", "joint_baseline", "baseline"),
        ("joint", "joint_bioaware", "bioaware"),
    ):
        checkpoint_path = checkpoint_root / checkpoint_name / "best.pth"
        checkpoint, state = checkpoint_state(checkpoint_path, device)
        assert_checkpoint(checkpoint, model_variant, joint_model=True)

        first_record = read_records(sequence_path(data_root, GENES[0]))[0]
        first_graphs, _ = build_graphs(first_record, model_variant)
        input_dimension = int(first_graphs[0].x.shape[1])
        model, _ = joint.build_model(
            model_variant,
            input_dimension=input_dimension,
            output_dimension=len(JOINT_TISSUES),
            device=device,
        )
        load_model_state(model, state)

        print(f"[inference] {output_variant}: {checkpoint_path}")
        for gene in GENES:
            record = read_records(sequence_path(data_root, gene))[0]
            graphs, metadata = build_graphs(record, model_variant)
            probabilities = infer(
                model,
                graphs,
                device,
                model_kind="joint",
                tissue_columns=selected_columns,
                batch_size=batch_size,
            )
            for column, tissue in enumerate(TISSUES):
                levels = load_editing_levels(table_path(data_root, gene, tissue))
                write_predictions(
                    output_path(out_root, output_variant, gene, tissue),
                    record,
                    metadata,
                    probabilities[:, column],
                    levels,
                )


def write_metadata(path: Path, device: torch.device):
    payload = {
        "analysis": "coding-target out-of-domain inference",
        "device": str(device),
        "torch_version": torch.__version__,
        "variants": ["baseline", "bioaware", "joint_baseline", "joint"],
        "genes": list(GENES),
        "tissues": list(TISSUES),
        "inference_only": True,
        "training_or_fine_tuning": False,
        "coding_labels_used_as_model_inputs": False,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PACKAGE / "data")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=PACKAGE / "checkpoints"
    )
    parser.add_argument(
        "--output-root", type=Path, default=PACKAGE / "predictions"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses CUDA when available and otherwise CPU",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.manual_seed(42)
    np.random.seed(42)
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")
    run_single_models(
        args.data_root,
        args.checkpoint_root,
        args.output_root,
        device,
        args.batch_size,
    )
    run_joint_models(
        args.data_root,
        args.checkpoint_root,
        args.output_root,
        device,
        args.batch_size,
    )
    write_metadata(args.output_root / "inference_metadata.json", device)
    print(f"[done] predictions: {args.output_root}")


if __name__ == "__main__":
    main()
