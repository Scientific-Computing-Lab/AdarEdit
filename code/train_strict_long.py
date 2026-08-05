#!/usr/bin/env python3
"""Train one per-context model with validation-only selection."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_DISABLE_NNPACK", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch_geometric.loader import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

import bioaware_gnn as bio  # noqa: E402
import gnnadar_verb_compact as baseline  # noqa: E402

GRAPH_VERSION = "graph_v2"
BIO_ABLATIONS = (
    "full",
    "no_neighbors",
    "no_geometry",
    "no_seq_branch",
    "untyped_edges",
    "no_pair_edges",
    "shuffle_pair_edges",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "bioaware"), required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=PACKAGE_ROOT / "cache/single")
    parser.add_argument("--out-root", type=Path, default=PACKAGE_ROOT / "runs")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help=(
            "Checkpoint-only training chunk boundary. Training stops after this "
            "epoch without evaluating test unless it equals --epochs."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument(
        "--bio-ablation",
        choices=BIO_ABLATIONS,
        default="full",
        help="Bio-aware component ablation; full is the primary reported model.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def configure(seed: int, threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, threads)))
    except RuntimeError:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_load(path: Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(payload, path: Path) -> None:
    """Write a checkpoint atomically so a killed job cannot corrupt it."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_row_indices(path: Path) -> list[int]:
    rows = []
    with path.open() as handle:
        for index, line in enumerate(handle):
            sequence, structure, _, _ = baseline.parse_openai_json(json.loads(line))
            if sequence and structure and len(sequence) == len(structure):
                rows.append(index)
    return rows


def transform_bioaware_graph(graph, ablation: str, seed: int):
    """Apply one deterministic component ablation to a bio-aware graph."""
    if ablation in ("full", "no_neighbors", "no_geometry", "no_seq_branch"):
        return graph
    transformed = copy.deepcopy(graph)
    is_sequence = transformed.edge_scalar[:, 0] > 0.5
    if ablation == "untyped_edges":
        transformed.edge_type = torch.zeros_like(transformed.edge_type)
        return transformed
    if ablation == "no_pair_edges":
        transformed.edge_index = transformed.edge_index[:, is_sequence].contiguous()
        transformed.edge_type = transformed.edge_type[is_sequence].contiguous()
        transformed.edge_scalar = transformed.edge_scalar[is_sequence].contiguous()
        return transformed
    if ablation != "shuffle_pair_edges":
        raise ValueError(f"Unsupported bio-aware ablation: {ablation}")

    sequence_index = transformed.edge_index[:, is_sequence]
    sequence_type = transformed.edge_type[is_sequence]
    sequence_scalar = transformed.edge_scalar[is_sequence]
    pair_edges = transformed.edge_index[:, ~is_sequence].t().tolist()
    original_pairs = sorted(
        {tuple(sorted(edge)) for edge in pair_edges if edge[0] != edge[1]}
    )
    paired_nodes = sorted({node for pair in original_pairs for node in pair})
    if len(paired_nodes) < 4:
        return transformed
    generator = random.Random(seed + int(transformed.x.shape[0]))
    shuffled_pairs = []
    for _ in range(64):
        permutation = paired_nodes[:]
        generator.shuffle(permutation)
        candidate = [
            tuple(sorted((permutation[index], permutation[index + 1])))
            for index in range(0, len(permutation), 2)
        ]
        if (
            set(candidate).isdisjoint(set(original_pairs))
            and all(abs(left - right) > 1 for left, right in candidate)
        ):
            shuffled_pairs = candidate
            break
    if not shuffled_pairs:
        raise RuntimeError(
            "Could not create a shuffled matching disjoint from both original "
            "pair edges and adjacent backbone relations."
        )

    edge_index = sequence_index.t().tolist()
    edge_type = sequence_type.tolist()
    edge_scalar = sequence_scalar.tolist()
    for left, right in shuffled_pairs:
        base_left = bio.BASES[int(transformed.seq_ids[left])]
        base_right = bio.BASES[int(transformed.seq_ids[right])]
        kind = bio.classify_pair(base_left, base_right)
        compatibility = float(
            bio.PAIR_COMPATIBILITY.get((base_left, base_right), 0.0)
        )
        # Edge geometry remains the endpoint geometry from the original graph;
        # only the identity of the connected partner is randomized.
        scalar_lr = [
            0.0,
            1.0,
            0.0,
            float(transformed.stem_raw[left]),
            float(transformed.stem_raw[right]),
            float(transformed.x[left, 19]),
            float(transformed.x[right, 19]),
            float(transformed.x[left, 20]),
            float(transformed.x[right, 20]),
            compatibility,
            abs(float(transformed.x[left, 6])),
            abs(float(transformed.x[right, 6])),
        ]
        scalar_rl = [
            scalar_lr[0],
            scalar_lr[1],
            scalar_lr[2],
            scalar_lr[4],
            scalar_lr[3],
            scalar_lr[6],
            scalar_lr[5],
            scalar_lr[8],
            scalar_lr[7],
            scalar_lr[9],
            scalar_lr[11],
            scalar_lr[10],
        ]
        edge_index.extend([[left, right], [right, left]])
        edge_type.extend([kind, kind])
        edge_scalar.extend([scalar_lr, scalar_rl])
    transformed.edge_index = torch.tensor(
        edge_index, dtype=torch.long
    ).t().contiguous()
    transformed.edge_type = torch.tensor(edge_type, dtype=torch.long)
    transformed.edge_scalar = torch.tensor(edge_scalar, dtype=torch.float)
    return transformed


def load_or_build_graphs(
    source: Path, cache: Path, variant: str, bio_ablation: str = "full", seed: int = 42
) -> tuple[list, list[int]]:
    rows_path = cache.with_suffix(".rows.json")
    manifest_path = cache.with_suffix(".manifest.json")
    expected = {
        "graph_version": GRAPH_VERSION,
        "variant": variant,
        "bio_ablation": bio_ablation,
        "ablation_seed": seed if bio_ablation == "shuffle_pair_edges" else None,
        "source_sha256": sha256(source),
    }
    if cache.exists() and rows_path.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text()) == expected:
            return torch_load(cache), json.loads(rows_path.read_text())
    if variant == "baseline":
        graphs = baseline.load_jsonl_to_graphs(source)
    else:
        use_neighbors = bio_ablation != "no_neighbors"
        use_geometry = bio_ablation != "no_geometry"
        graphs = bio.load_graphs(
            str(source),
            use_neighbors=use_neighbors,
            use_geometry=use_geometry,
            plfold_dir=None,
        )
        graphs = [
            transform_bioaware_graph(graph, bio_ablation, seed + index)
            for index, graph in enumerate(graphs)
        ]
    rows = valid_row_indices(source)
    if len(graphs) != len(rows):
        raise RuntimeError(
            f"Graph/row mismatch for {source}: {len(graphs)} versus {len(rows)}"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, cache)
    rows_path.write_text(json.dumps(rows) + "\n")
    manifest_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    return graphs, rows


def metric_values(labels, probabilities, threshold) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    has_both = len(np.unique(labels)) > 1
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(
            recall_score(labels, predictions, pos_label=0, zero_division=0)
        ),
        "auc": float(roc_auc_score(labels, probabilities))
        if has_both
        else float("nan"),
        "auprc": float(average_precision_score(labels, probabilities))
        if has_both
        else float("nan"),
        "threshold": float(threshold),
        "n": int(len(labels)),
        "positive_n": int(labels.sum()),
    }


def choose_threshold(labels, probabilities) -> dict[str, float]:
    candidates = [
        metric_values(labels, probabilities, float(threshold))
        for threshold in np.linspace(0.1, 0.9, 33)
    ]
    return max(candidates, key=lambda item: item["f1"])


def create_model(
    variant: str,
    in_dim: int,
    device: torch.device,
    bio_ablation: str = "full",
):
    if variant == "baseline":
        model = baseline.RNAEditingGNN(8, 32, 1, 4).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        loss_function = nn.BCELoss()
    else:
        model = bio.BioAwareGNN(
            in_dim=in_dim,
            hidden=96,
            heads=4,
            layers=3,
            edge_emb_dim=6,
            edge_scalar_dim=12,
            dropout=0.1,
            seq_branch_dim=0 if bio_ablation == "no_seq_branch" else 128,
            use_global_attn=False,
            global_attn_heads=4,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=3e-3, weight_decay=1e-4
        )
        loss_function = nn.BCEWithLogitsLoss()
    return model, optimizer, loss_function


def evaluate(variant, model, loader, device):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if variant == "baseline":
                output, _ = model(batch)
            else:
                output = torch.sigmoid(model(batch))
            probabilities.extend(output.detach().cpu().view(-1).tolist())
            labels.extend(batch.y.detach().cpu().view(-1).int().tolist())
    return np.asarray(probabilities, dtype=float), np.asarray(labels, dtype=int)


def train_epoch(
    variant, model, loader, optimizer, loss_function, device, grad_clip
):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        if variant == "baseline":
            output, _ = model(batch)
        else:
            output = model(batch)
        loss = loss_function(output.view(-1), batch.y.view(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {float(loss)}")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def rng_state() -> dict:
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()
    return value


def restore_rng_state(value: dict) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    # Loading a checkpoint with map_location=<CUDA device> also moves the
    # serialized RNG tensors to CUDA.  PyTorch's RNG restoration APIs require
    # CPU ByteTensors, even when restoring CUDA-generator states.
    torch_cpu_state = value["torch_cpu"].detach().to(
        device="cpu", dtype=torch.uint8
    )
    torch.set_rng_state(torch_cpu_state)
    if torch.cuda.is_available() and "torch_cuda" in value:
        torch_cuda_states = [
            state.detach().to(device="cpu", dtype=torch.uint8)
            for state in value["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(torch_cuda_states)


def checkpoint_payload(
    epoch, args, model, optimizer, best, config, *, resumable=True
):
    payload = {
        "epoch": epoch,
        "variant": args.variant,
        "tissue": args.context,
        "context": args.context,
        "selection_split": "valid",
        "graph_version": GRAPH_VERSION,
        "model_state": model.state_dict(),
        "best": best,
        "config": config,
    }
    if resumable:
        payload["optimizer_state"] = optimizer.state_dict()
        payload["rng_state"] = rng_state()
    return payload


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "selection_split",
    "selection_acc",
    "selection_auc",
    "selection_auprc",
    "selection_f1",
    "selection_precision",
    "selection_recall",
    "selection_specificity",
    "selection_threshold",
    "best_so_far",
    "best_epoch_so_far",
    "elapsed_sec",
]


def append_history(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def reconcile_history(path: Path, completed_epoch: int) -> None:
    if not path.exists():
        return
    with path.open() as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["epoch"]) <= completed_epoch
        ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def export_predictions(
    split_dir: Path,
    valid_rows: list[int],
    probabilities,
    labels,
    threshold,
    variant,
    context,
    path: Path,
) -> None:
    metadata_path = split_dir / "test.metadata.csv"
    if metadata_path.exists():
        table = pd.read_csv(metadata_path).iloc[valid_rows].copy()
    else:
        table = pd.DataFrame({"source_row_index": valid_rows})
    if len(table) != len(probabilities):
        raise RuntimeError(
            f"Prediction/table mismatch: {len(probabilities)} versus {len(table)}"
        )
    table["prob"] = probabilities
    table["label_from_loader"] = labels
    table["pred_label"] = (probabilities >= threshold).astype(int)
    table["threshold"] = threshold
    table["variant"] = variant
    table["tissue"] = context
    table["graph_version"] = GRAPH_VERSION
    table.to_csv(path, index=False)


def save_curves(path: Path, probabilities, labels) -> None:
    thresholds = np.linspace(0.0, 1.0, 101)
    precision, recall, true_positive_rate, false_positive_rate = [], [], [], []
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        tp = int(((predicted == 1) & (labels == 1)).sum())
        fp = int(((predicted == 1) & (labels == 0)).sum())
        fn = int(((predicted == 0) & (labels == 1)).sum())
        tn = int(((predicted == 0) & (labels == 0)).sum())
        precision.append(tp / max(1, tp + fp))
        recall.append(tp / max(1, tp + fn))
        true_positive_rate.append(tp / max(1, tp + fn))
        false_positive_rate.append(fp / max(1, fp + tn))
    np.savez_compressed(
        path,
        thresholds=thresholds,
        precision=precision,
        recall=recall,
        tpr=true_positive_rate,
        fpr=false_positive_rate,
        probs=probabilities,
        labels=labels,
    )


def environment(device: torch.device) -> dict:
    import sklearn
    import torch_geometric

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
    }


def main() -> None:
    args = parse_args()
    run_until_epoch = (
        args.epochs
        if args.stop_after_epoch is None
        else args.stop_after_epoch
    )
    if not 1 <= run_until_epoch <= args.epochs:
        raise ValueError(
            "--stop-after-epoch must be between 1 and --epochs inclusive"
        )
    if args.variant == "baseline" and args.bio_ablation != "full":
        raise ValueError("--bio-ablation applies only to --variant bioaware")
    configure(args.seed, args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA not detected; pass --allow-cpu only for testing.")

    split_dir = args.data_root / args.context
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        if not (split_dir / name).is_file():
            raise FileNotFoundError(split_dir / name)
    run_dir = (
        args.out_root / f"{args.variant}_{args.context}"
        if args.bio_ablation == "full"
        else args.out_root / "ablations" / args.context / args.bio_ablation
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        args.cache_root / args.variant / args.context
        if args.bio_ablation == "full"
        else args.cache_root / "ablations" / args.context / args.bio_ablation
    )

    config = {
        **vars(args),
        "data_root": str(args.data_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "out_root": str(args.out_root.resolve()),
        "device": str(device),
        "graph_version": GRAPH_VERSION,
        "selection_split": "valid",
        "test_used_during_training": False,
        "baseline_pair_edge_multiplicity_per_direction": 1,
        "stem_feature": "true_contiguous_integer_stem_length",
        "sequential_edge_pair_compatibility": 0.0,
        "t_and_u_share_one_hot_channel": True,
        "bio_ablation": args.bio_ablation,
        "gradient_clip_note": (
            "Gradient clipping is disabled when gradient_clip is 0.0."
        ),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(environment(device), indent=2, sort_keys=True) + "\n"
    )

    graphs, rows = {}, {}
    for split in ("train", "valid", "test"):
        graphs[split], rows[split] = load_or_build_graphs(
            split_dir / f"{split}.jsonl",
            cache_dir / f"{split}.pt",
            args.variant,
            args.bio_ablation,
            args.seed + {"train": 0, "valid": 1_000_000, "test": 2_000_000}[split],
        )
    loaders = {
        "train": DataLoader(
            graphs["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        ),
        "valid": DataLoader(
            graphs["valid"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
        "test": DataLoader(
            graphs["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
    }
    model, optimizer, loss_function = create_model(
        args.variant,
        int(graphs["train"][0].x.shape[1]),
        device,
        args.bio_ablation,
    )
    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"
    history_path = run_dir / "history.csv"
    start_epoch = 1
    best = {"f1": -1.0, "epoch": 0, "metrics": None}
    if args.resume and last_path.exists():
        saved = torch_load(last_path, map_location=device)
        if saved.get("graph_version") != GRAPH_VERSION:
            raise RuntimeError("Refusing to resume from another graph version.")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        best = saved["best"]
        start_epoch = int(saved["epoch"]) + 1
        restore_rng_state(saved["rng_state"])
        reconcile_history(history_path, int(saved["epoch"]))

    for epoch in range(start_epoch, run_until_epoch + 1):
        started = time.time()
        train_loss = train_epoch(
            args.variant,
            model,
            loaders["train"],
            optimizer,
            loss_function,
            device,
            args.grad_clip,
        )
        valid_probabilities, valid_labels = evaluate(
            args.variant, model, loaders["valid"], device
        )
        metrics = choose_threshold(valid_labels, valid_probabilities)
        improved = metrics["f1"] > best["f1"] + 1e-6
        if improved:
            best = {"f1": metrics["f1"], "epoch": epoch, "metrics": metrics}
            atomic_torch_save(
                checkpoint_payload(
                    epoch,
                    args,
                    model,
                    optimizer,
                    best,
                    config,
                    resumable=False,
                ),
                best_path,
            )
        append_history(
            history_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "selection_split": "valid",
                "selection_acc": metrics["acc"],
                "selection_auc": metrics["auc"],
                "selection_auprc": metrics["auprc"],
                "selection_f1": metrics["f1"],
                "selection_precision": metrics["precision"],
                "selection_recall": metrics["recall"],
                "selection_specificity": metrics["specificity"],
                "selection_threshold": metrics["threshold"],
                "best_so_far": int(improved),
                "best_epoch_so_far": int(best["epoch"]),
                "elapsed_sec": time.time() - started,
            },
        )
        payload = checkpoint_payload(epoch, args, model, optimizer, best, config)
        atomic_torch_save(payload, last_path)
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            atomic_torch_save(
                payload, checkpoint_dir / f"epoch_{epoch:04d}.pth"
            )
        print(
            f"[{args.variant}][{args.context}] epoch={epoch}/{args.epochs} "
            f"loss={train_loss:.6f} valid_f1={metrics['f1']:.6f} "
            f"valid_auc={metrics['auc']:.6f} threshold={metrics['threshold']:.3f} "
            f"best_epoch={best['epoch']}",
            flush=True,
        )

    if run_until_epoch < args.epochs:
        if not last_path.is_file():
            raise RuntimeError("Checkpoint-only chunk produced no last.pth")
        print(
            f"[{args.variant}][{args.context}] CHECKPOINT_ONLY "
            f"completed_through_epoch={run_until_epoch} "
            f"final_epoch={args.epochs}; test split not evaluated",
            flush=True,
        )
        return

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was produced.")
    saved = torch_load(best_path, map_location=device)
    model.load_state_dict(saved["model_state"])
    selected = saved["best"]["metrics"]
    test_probabilities, test_labels = evaluate(
        args.variant, model, loaders["test"], device
    )
    threshold = float(selected["threshold"])
    test_metrics = metric_values(test_labels, test_probabilities, threshold)
    export_predictions(
        split_dir,
        rows["test"],
        test_probabilities,
        test_labels,
        threshold,
        args.variant,
        args.context,
        run_dir / "test_predictions.csv",
    )
    with (run_dir / "test_predictions.jsonl").open("w") as handle:
        for probability, label in zip(test_probabilities, test_labels):
            handle.write(
                json.dumps(
                    {"prob": float(probability), "label": int(label)}
                )
                + "\n"
            )
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        probs=test_probabilities,
        probabilities=test_probabilities,
        labels=test_labels,
        threshold=threshold,
    )
    save_curves(run_dir / "test_curves.npz", test_probabilities, test_labels)

    summary = {
        "variant": args.variant,
        "tissue": args.context,
        "context": args.context,
        "graph_version": GRAPH_VERSION,
        "bio_ablation": args.bio_ablation,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch": int(saved["best"]["epoch"]),
        "selection_split": "valid",
        "test_used_during_training": False,
        "valid_selection_metrics": selected,
        "test_selection_metrics": None,
        "test_metrics": test_metrics,
        "sizes": {key: len(value) for key, value in graphs.items()},
        "data_sha256": {
            split: sha256(split_dir / f"{split}.jsonl")
            for split in ("train", "valid", "test")
        },
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "history_csv": str(history_path.resolve()),
            "best_checkpoint": str(best_path.resolve()),
            "latest_checkpoint": str(last_path.resolve()),
            "predictions_csv": str(
                (run_dir / "test_predictions.csv").resolve()
            ),
            "predictions_jsonl": str(
                (run_dir / "test_predictions.jsonl").resolve()
            ),
            "curves_npz": str((run_dir / "test_curves.npz").resolve()),
            "graph_cache_dir": str(cache_dir.resolve()),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "COMPLETE").write_text(
        f"{GRAPH_VERSION} epochs_completed={args.epochs} "
        f"best_epoch={summary['best_epoch']}\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
