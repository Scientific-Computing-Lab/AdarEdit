#!/usr/bin/env python3
"""Train joint multi-label models with or without the Combined output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
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

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

import bioaware_gnn as bio  # noqa: E402
import gnnadar_verb_compact as baseline_graph  # noqa: E402
from joint_baseline_gnn import JointBaselineGNN  # noqa: E402

ALL_TISSUES = ["Artery", "Brain", "Combined", "Liver", "MuscleSkeletal"]
NO_COMBINED_TISSUES = ["Artery", "Brain", "Liver", "MuscleSkeletal"]
GRAPH_VERSION = "graph_v2"


class JointBioAwareGNN(bio.BioAwareGNN):
    """Checkpoint-compatible bio-aware encoder with a configurable-width head."""

    def __init__(self, *args, out_dim: int, **kwargs):
        super().__init__(*args, **kwargs)
        fusion = self.head[0].in_features
        hidden = self.head[0].out_features
        dropout = self.head[2].p
        self.head = nn.Sequential(
            nn.Linear(fusion, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def encode(self, data):
        embedded_type = self.edge_emb(data.edge_type)
        edge_attributes = self.edge_mlp(
            torch.cat([embedded_type, data.edge_scalar], dim=1)
        )
        features = data.x
        for convolution, batch_norm in zip(self.convs, self.bns):
            features = convolution(
                features, data.edge_index, edge_attr=edge_attributes
            )
            features = torch.relu(features)
            features = batch_norm(features)
            features = self.dropout(features)
        graph_embedding = bio.global_mean_pool(features, data.batch)
        sequence_batch, mask = self._build_seq_batch(
            data.seq_ids, data.batch
        )
        sequence_embedding = self.seq_encoder(sequence_batch, mask)
        return torch.cat([graph_embedding, sequence_embedding], dim=1)

    def forward(self, data):
        return self.head(self.encode(data))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "bioaware"), required=True)
    parser.add_argument(
        "--tissue-set", choices=("with_combined", "no_combined"), required=True
    )
    parser.add_argument(
        "--data-root", type=Path, default=REPOSITORY_ROOT / "data/human"
    )
    parser.add_argument(
        "--cache-root", type=Path, default=ANALYSIS_ROOT / "outputs/cache"
    )
    parser.add_argument(
        "--out-root", type=Path, default=ANALYSIS_ROOT / "outputs/runs"
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help=(
            "Checkpoint-only Slurm chunk boundary. Training stops after this "
            "epoch without evaluating test unless it equals --epochs."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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


def parse_line(line: str):
    record = json.loads(line)
    sequence, structure, target, label = baseline_graph.parse_openai_json(record)
    if not sequence or not structure or len(sequence) != len(structure):
        return None
    return (sequence.replace("T", "U"), structure, target), label


def read_split(data_root: Path, split: str, tissues: list[str]):
    sites = OrderedDict()
    for tissue in tissues:
        path = data_root / tissue / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open() as handle:
            for line in handle:
                parsed = parse_line(line)
                if parsed is None:
                    continue
                key, label = parsed
                sites.setdefault(key, {})[tissue] = label
    return sites


def build_graph(key, variant):
    sequence, structure, target = key
    if variant == "baseline":
        return baseline_graph.create_rna_graph(
            sequence, structure, target, label=0
        )
    return bio.build_graph(
        bio.GraphExample(sequence, structure, target, 0),
        use_neighbors=True,
        use_geometry=True,
        plfold_dir=None,
    )


def make_graphs(sites, variant, tissues):
    tissue_to_index = {tissue: index for index, tissue in enumerate(tissues)}
    graphs = []
    for key, labels in sites.items():
        graph = build_graph(key, variant)
        graph.y = torch.zeros(1, len(tissues))
        graph.y_mask = torch.zeros(1, len(tissues))
        for tissue, value in labels.items():
            index = tissue_to_index[tissue]
            graph.y[0, index] = float(value)
            graph.y_mask[0, index] = 1.0
        graphs.append(graph)
    return graphs


def load_or_build(
    data_root, cache_root, split, variant, tissue_set, tissues
):
    cache = cache_root / tissue_set / variant / f"{split}.pt"
    manifest_path = cache.with_suffix(".manifest.json")
    source_hashes = {
        tissue: sha256(data_root / tissue / f"{split}.jsonl")
        for tissue in tissues
    }
    expected = {
        "graph_version": GRAPH_VERSION,
        "variant": variant,
        "tissue_set": tissue_set,
        "tissues": tissues,
        "source_sha256": source_hashes,
    }
    if cache.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text()) == expected:
            return torch_load(cache)
    sites = read_split(data_root, split, tissues)
    graphs = make_graphs(sites, variant, tissues)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, cache)
    manifest_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    return graphs


def build_model(variant, input_dimension, output_dimension, device):
    if variant == "baseline":
        model = JointBaselineGNN(
            input_dim=input_dimension,
            hidden_dim=32,
            output_dim=output_dimension,
            num_heads=4,
            num_layers=3,
            dropout_rate=0.2,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
    else:
        model = JointBioAwareGNN(
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
            out_dim=output_dimension,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=3e-3, weight_decay=1e-4
        )
    return model, optimizer


def binary_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    has_both = len(np.unique(labels)) > 1
    return {
        "n": int(len(labels)),
        "acc": float(accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(
            precision_score(labels, predicted, zero_division=0)
        ),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities))
        if has_both
        else float("nan"),
        "auprc": float(average_precision_score(labels, probabilities))
        if has_both
        else float("nan"),
        "threshold": float(threshold),
    }


@torch.no_grad()
def predict(model, loader, device, output_dimension):
    model.eval()
    probabilities, labels, masks = [], [], []
    for batch in loader:
        batch = batch.to(device)
        probabilities.append(
            torch.sigmoid(model(batch)).cpu().numpy().reshape(-1, output_dimension)
        )
        labels.append(batch.y.cpu().numpy().reshape(-1, output_dimension))
        masks.append(batch.y_mask.cpu().numpy().reshape(-1, output_dimension))
    return np.concatenate(probabilities), np.concatenate(labels), np.concatenate(masks)


def choose_thresholds(probabilities, labels, masks, tissues):
    thresholds = {}
    for index, tissue in enumerate(tissues):
        observed = masks[:, index] > 0
        tissue_labels = labels[observed, index]
        tissue_probabilities = probabilities[observed, index]
        candidates = np.linspace(0.1, 0.9, 33)
        thresholds[tissue] = float(
            max(
                candidates,
                key=lambda threshold: f1_score(
                    tissue_labels,
                    (tissue_probabilities >= threshold).astype(int),
                    zero_division=0,
                ),
            )
        )
    return thresholds


def per_tissue_metrics(probabilities, labels, masks, tissues, thresholds):
    output = {}
    for index, tissue in enumerate(tissues):
        observed = masks[:, index] > 0
        output[tissue] = binary_metrics(
            labels[observed, index],
            probabilities[observed, index],
            thresholds[tissue],
        )
    macro = {
        metric: float(np.mean([output[tissue][metric] for tissue in tissues]))
        for metric in ("f1", "auc", "auprc")
    }
    return output, macro


def all_metrics(probabilities, labels, masks, tissues, thresholds):
    by_tissue, macro = per_tissue_metrics(
        probabilities, labels, masks, tissues, thresholds
    )
    observed = masks > 0
    threshold_grid = np.asarray(
        [[thresholds[tissue] for tissue in tissues]] * len(probabilities)
    )
    pooled_labels = labels[observed]
    pooled_probabilities = probabilities[observed]
    pooled_predictions = (
        pooled_probabilities >= threshold_grid[observed]
    ).astype(int)
    pooled = {
        "n": int(len(pooled_labels)),
        "acc": float(accuracy_score(pooled_labels, pooled_predictions)),
        "f1": float(
            f1_score(pooled_labels, pooled_predictions, zero_division=0)
        ),
        "precision": float(
            precision_score(
                pooled_labels, pooled_predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(pooled_labels, pooled_predictions, zero_division=0)
        ),
        "auc": float(roc_auc_score(pooled_labels, pooled_probabilities)),
        "auprc": float(
            average_precision_score(pooled_labels, pooled_probabilities)
        ),
    }
    return {
        "test_macro_f1": macro["f1"],
        "test_macro_auc": macro["auc"],
        "test_macro_auprc": macro["auprc"],
        "test_pooled": pooled,
        "test_by_tissue": by_tissue,
    }


def train_epoch(model, loader, optimizer, device, output_dimension, grad_clip):
    model.train()
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        labels = batch.y.view(-1, output_dimension)
        mask = batch.y_mask.view(-1, output_dimension)
        loss = (loss_function(logits, labels) * mask).sum() / mask.sum().clamp(1)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite joint loss: {float(loss)}")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def rng_state():
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()
    return value


def restore_rng_state(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    # map_location=<CUDA device> is appropriate for model/optimizer tensors,
    # but PyTorch RNG restoration requires CPU ByteTensors.
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


def checkpoint(
    epoch, args, model, optimizer, best, config, tissues, *, resumable=True
):
    payload = {
        "epoch": epoch,
        "variant": args.variant,
        "mode": "multilabel",
        "tissue_set": args.tissue_set,
        "tissues": tissues,
        "graph_version": GRAPH_VERSION,
        "model_state": model.state_dict(),
        "best": best,
        "config": config,
    }
    if resumable:
        payload["optimizer_state"] = optimizer.state_dict()
        payload["rng_state"] = rng_state()
    return payload


def append_history(path, row):
    fields = list(row)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def reconcile_history(path, completed_epoch):
    if not path.exists():
        return
    with path.open() as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["epoch"]) <= completed_epoch
        ]
    fields = [
        "epoch",
        "train_loss",
        "valid_macro_f1",
        "valid_macro_auc",
        "valid_macro_auprc",
        "best_epoch",
        "elapsed_sec",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    configure(args.seed, args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA not detected; pass --allow-cpu only for testing.")
    tissues = (
        ALL_TISSUES
        if args.tissue_set == "with_combined"
        else NO_COMBINED_TISSUES
    )
    suffix = "" if args.tissue_set == "with_combined" else "_noCombined"
    run_name = f"joint_{args.variant}{suffix}"
    run_dir = args.out_root / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = {
        **vars(args),
        "data_root": str(args.data_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "out_root": str(args.out_root.resolve()),
        "device": str(device),
        "mode": "multilabel",
        "tissues": tissues,
        "num_outputs": len(tissues),
        "combined_included": "Combined" in tissues,
        "graph_version": GRAPH_VERSION,
        "selection_split": "valid",
        "test_used_during_training": False,
        "baseline_pair_edge_multiplicity_per_direction": 1,
        "stem_feature": "true_contiguous_integer_stem_length",
        "sequential_edge_pair_compatibility": 0.0,
        "t_and_u_share_one_hot_channel": True,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n"
    )

    graphs = {
        split: load_or_build(
            args.data_root,
            args.cache_root,
            split,
            args.variant,
            args.tissue_set,
            tissues,
        )
        for split in ("train", "valid", "test")
    }
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
    output_dimension = len(tissues)
    model, optimizer = build_model(
        args.variant,
        int(graphs["train"][0].x.shape[1]),
        output_dimension,
        device,
    )
    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"
    history_path = run_dir / "history.csv"
    best = {"macro_f1": -1.0, "epoch": 0}
    start_epoch = 1
    if args.resume and last_path.exists():
        saved = torch_load(last_path, map_location=device)
        if (
            saved.get("graph_version") != GRAPH_VERSION
            or saved.get("tissues") != tissues
        ):
            raise RuntimeError("Refusing incompatible joint resume checkpoint.")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        best = saved["best"]
        start_epoch = int(saved["epoch"]) + 1
        restore_rng_state(saved["rng_state"])
        reconcile_history(history_path, int(saved["epoch"]))

    for epoch in range(start_epoch, run_until_epoch + 1):
        started = time.time()
        loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            output_dimension,
            args.grad_clip,
        )
        probabilities, labels, masks = predict(
            model, loaders["valid"], device, output_dimension
        )
        thresholds = choose_thresholds(
            probabilities, labels, masks, tissues
        )
        _, macro = per_tissue_metrics(
            probabilities, labels, masks, tissues, thresholds
        )
        improved = macro["f1"] > best["macro_f1"] + 1e-6
        if improved:
            best = {
                "macro_f1": macro["f1"],
                "macro_auc": macro["auc"],
                "macro_auprc": macro["auprc"],
                "epoch": epoch,
                "thresholds": thresholds,
            }
            atomic_torch_save(
                checkpoint(
                    epoch,
                    args,
                    model,
                    optimizer,
                    best,
                    config,
                    tissues,
                    resumable=False,
                ),
                best_path,
            )
        append_history(
            history_path,
            {
                "epoch": epoch,
                "train_loss": loss,
                "valid_macro_f1": macro["f1"],
                "valid_macro_auc": macro["auc"],
                "valid_macro_auprc": macro["auprc"],
                "best_epoch": best["epoch"],
                "elapsed_sec": time.time() - started,
            },
        )
        payload = checkpoint(
            epoch, args, model, optimizer, best, config, tissues
        )
        atomic_torch_save(payload, last_path)
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            atomic_torch_save(
                payload, checkpoint_dir / f"epoch_{epoch:04d}.pth"
            )
        print(
            f"[joint][{args.variant}][{args.tissue_set}] "
            f"epoch={epoch}/{args.epochs} loss={loss:.6f} "
            f"valid_macro_f1={macro['f1']:.6f} "
            f"valid_macro_auc={macro['auc']:.6f} best_epoch={best['epoch']}",
            flush=True,
        )

    if run_until_epoch < args.epochs:
        if not last_path.is_file():
            raise RuntimeError("Checkpoint-only joint chunk produced no last.pth")
        print(
            f"[joint][{args.variant}][{args.tissue_set}] CHECKPOINT_ONLY "
            f"completed_through_epoch={run_until_epoch} "
            f"final_epoch={args.epochs}; test split not evaluated",
            flush=True,
        )
        return

    if not best_path.exists():
        raise RuntimeError("No best joint checkpoint was produced.")
    saved = torch_load(best_path, map_location=device)
    model.load_state_dict(saved["model_state"])
    thresholds = saved["best"]["thresholds"]
    probabilities, labels, masks = predict(
        model, loaders["test"], device, output_dimension
    )
    metrics = all_metrics(
        probabilities, labels, masks, tissues, thresholds
    )
    summary = {
        "variant": args.variant,
        "mode": "multilabel",
        "experiment": (
            "graph_with_combined"
            if args.tissue_set == "with_combined"
            else "graph_no_combined"
        ),
        "tissue_set": args.tissue_set,
        "tissues": tissues,
        "num_outputs": output_dimension,
        "combined_included": "Combined" in tissues,
        "graph_version": GRAPH_VERSION,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch": int(saved["best"]["epoch"]),
        "thresholds_from_valid": thresholds,
        **metrics,
        "sizes": {split: len(values) for split, values in graphs.items()},
        "observed_labels": {
            split: int(
                sum(float(graph.y_mask.sum()) for graph in values)
            )
            for split, values in graphs.items()
        },
        "selection_split": "valid",
        "test_used_during_training": False,
        "threshold_note": (
            "Per-output F1-optimal thresholds selected on validation only and "
            "applied unchanged to held-out test; AUROC/AUPRC are threshold-free."
        ),
        "leakage_note": (
            "The supplied global pair-disjoint split is reused across tissues."
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        run_dir / "test_curves.npz",
        probs=probabilities,
        labels=labels,
        mask=masks,
        tissues=np.asarray(tissues),
        thresholds=np.asarray([thresholds[tissue] for tissue in tissues]),
    )
    with (run_dir / "test_predictions.jsonl").open("w") as handle:
        for probability, label, mask in zip(probabilities, labels, masks):
            handle.write(
                json.dumps(
                    {
                        "prob": probability.tolist(),
                        "label": label.tolist(),
                        "mask": mask.tolist(),
                        "tissues": tissues,
                    }
                )
                + "\n"
            )
    (run_dir / "COMPLETE").write_text(
        f"{GRAPH_VERSION} epochs_completed={args.epochs} "
        f"best_epoch={summary['best_epoch']}\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
