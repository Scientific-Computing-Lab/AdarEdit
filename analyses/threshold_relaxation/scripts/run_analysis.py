#!/usr/bin/env python3
"""Score complete intermediate cohorts and regenerate threshold summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader


HERE = Path(__file__).resolve()
ANALYSIS = HERE.parents[1]
REPRO = HERE.parents[3]
CODE = REPRO / "code"
CONTEXTS = ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]
VARIANTS = ["baseline", "bioaware"]
# Checkpoint probabilities can vary slightly across supported PyTorch/PyG and
# CPU/GPU combinations. The graph labels must always match exactly.
VALIDATION_ATOL = 5e-4 if tuple(map(int, torch_geometric.__version__.split(".")[:2])) >= (2, 6) else 1e-3

sys.path.insert(0, str(CODE))
import bioaware_gnn as bio  # noqa: E402
import gnnadar_verb_compact as base  # noqa: E402
import train_strict_long as trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--validation-sites", type=int, default=64)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp"))
        / "adaredit_threshold_relaxation",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remap_baseline_state(state: dict, model: torch.nn.Module) -> dict:
    """Bridge the PyG 2.6 single-lin key to older local PyG when necessary."""
    model_keys = set(model.state_dict())
    if all(key in model_keys for key in state):
        return state
    remapped = {}
    for key, value in state.items():
        if key.endswith(".lin.weight"):
            prefix = key[: -len(".lin.weight")]
            source = f"{prefix}.lin_src.weight"
            destination = f"{prefix}.lin_dst.weight"
            if source in model_keys and destination in model_keys:
                remapped[source] = value.clone()
                remapped[destination] = value.clone()
                continue
        remapped[key] = value
    return remapped


def load_model(
    variant: str, context: str, device: torch.device, input_dim: int | None
) -> torch.nn.Module:
    checkpoint = REPRO / "checkpoints" / f"{variant}_{context}" / "best.pth"
    payload = torch.load(checkpoint, map_location=device)
    config = payload.get("config", {})
    required = {
        "variant": variant,
        "graph_version": "graph_v2",
        "selection_split": "valid",
        "test_used_during_training": False,
        "baseline_pair_edge_multiplicity_per_direction": 1,
        "stem_feature": "true_contiguous_integer_stem_length",
        "sequential_edge_pair_compatibility": 0.0,
        "t_and_u_share_one_hot_channel": True,
    }
    mismatches = {
        key: (expected, config.get(key))
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"{variant}/{context}: checkpoint mismatch {mismatches}")
    if variant == "baseline":
        model = base.RNAEditingGNN(8, 32, 1, 4)
        state = remap_baseline_state(payload["model_state"], model)
    else:
        if input_dim is None:
            raise ValueError("Bio-aware input dimension is required")
        model = bio.BioAwareGNN(
            in_dim=input_dim,
            hidden=96,
            heads=4,
            layers=3,
            edge_emb_dim=6,
            edge_scalar_dim=12,
            dropout=0.1,
            seq_branch_dim=128,
            use_global_attn=False,
            global_attn_heads=4,
        )
        state = payload["model_state"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def infer(
    variant: str,
    context: str,
    graphs: list,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    input_dim = int(graphs[0].x.shape[1]) if variant == "bioaware" else None
    model = load_model(variant, context, device, input_dim)
    scores = []
    labels = []
    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if variant == "baseline":
                probability, _ = model(batch)
                probability = probability.squeeze()
            else:
                probability = torch.sigmoid(model(batch)).squeeze()
            scores.extend(
                np.atleast_1d(probability.detach().cpu().numpy()).astype(float)
            )
            labels.extend(
                np.atleast_1d(batch.y.detach().cpu().numpy()).astype(int)
            )
    return np.asarray(scores), np.asarray(labels)


def cache_paths(cache: Path) -> list[Path]:
    return [
        cache,
        cache.with_suffix(".rows.json"),
        cache.with_suffix(".manifest.json"),
    ]


def clear_cache(cache: Path) -> None:
    for path in cache_paths(cache):
        path.unlink(missing_ok=True)


def build_graphs(source: Path, cache: Path, variant: str) -> tuple[list, list[int]]:
    return trainer.load_or_build_graphs(
        source=source,
        cache=cache,
        variant=variant,
        bio_ablation="full",
        seed=42,
    )


def validate_canonical_inference(args: argparse.Namespace, device: torch.device) -> list[dict]:
    validation_dir = args.work_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for context in CONTEXTS:
        source = REPRO / "data" / "human" / context / "test.jsonl"
        sample = validation_dir / f"{context}.jsonl"
        with source.open() as input_handle, sample.open("w") as output_handle:
            for index, line in enumerate(input_handle):
                if index >= args.validation_sites:
                    break
                output_handle.write(line)
        for variant in VARIANTS:
            cache = validation_dir / f"{variant}_{context}.pt"
            clear_cache(cache)
            graphs, rows = build_graphs(sample, cache, variant)
            if rows != list(range(len(graphs))):
                raise RuntimeError(f"{variant}/{context}: validation rows dropped")
            scores, labels = infer(
                variant,
                context,
                graphs,
                device,
                args.batch_size,
                args.num_workers,
            )
            reference = pd.read_csv(
                REPRO
                / "checkpoints"
                / f"{variant}_{context}"
                / "test_predictions.csv"
            ).iloc[: len(scores)]
            errors = np.abs(scores - reference["prob"].to_numpy(float))
            labels_match = np.array_equal(
                labels, reference["label_from_loader"].to_numpy(int)
            )
            maximum = float(errors.max())
            if not labels_match or not np.isfinite(scores).all() or maximum > VALIDATION_ATOL:
                raise RuntimeError(
                    f"{variant}/{context}: inference fidelity failed; "
                    f"labels={labels_match}, max_abs_error={maximum}"
                )
            results.append(
                {
                    "context": context,
                    "variant": variant,
                    "sites": len(scores),
                    "labels_match": labels_match,
                    "mean_abs_error": float(errors.mean()),
                    "max_abs_error": maximum,
                    "tolerance": VALIDATION_ATOL,
                    "status": "PASS",
                }
            )
            print(
                f"[fidelity] {variant}/{context}: max error={maximum:.6g}",
                flush=True,
            )
            del graphs
            clear_cache(cache)
    return results


def score_complete_cohorts(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    prediction_dir = ANALYSIS / "data" / "intermediate_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = args.work_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for context in CONTEXTS:
        levels = pd.read_csv(
            ANALYSIS / "data" / "cohorts" / f"inter_{context}_levels.csv"
        )["editing_level_pct"].to_numpy(float)
        source = ANALYSIS / "data" / "cohorts" / f"inter_{context}.jsonl"
        for variant in VARIANTS:
            output = prediction_dir / f"{variant}_{context}.csv"
            checkpoint = REPRO / "checkpoints" / f"{variant}_{context}" / "best.pth"
            if args.resume and output.exists():
                table = pd.read_csv(output)
                if len(table) == len(levels):
                    print(f"[resume] {variant}/{context}: {len(table):,}", flush=True)
                    write_prediction_metadata(
                        output, source, checkpoint, table, context, variant
                    )
                    completed.append(table)
                    continue
            cache = graph_dir / f"{variant}_{context}.pt"
            clear_cache(cache)
            graphs, rows = build_graphs(source, cache, variant)
            if rows != list(range(len(levels))):
                raise RuntimeError(f"{variant}/{context}: graph rows were dropped")
            scores, _ = infer(
                variant,
                context,
                graphs,
                device,
                args.batch_size,
                args.num_workers,
            )
            if len(scores) != len(levels):
                raise RuntimeError(f"{variant}/{context}: score/level count mismatch")
            table = pd.DataFrame(
                {
                    "tissue": context,
                    "variant": variant,
                    "editing_level": levels,
                    "score": scores,
                }
            )
            table.to_csv(output, index=False)
            write_prediction_metadata(
                output, source, checkpoint, table, context, variant
            )
            completed.append(table)
            print(
                f"[score] {variant}/{context}: n={len(table):,}, "
                f"mean={table['score'].mean():.4f}",
                flush=True,
            )
            del graphs
            clear_cache(cache)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    all_scores = pd.concat(completed, ignore_index=True)
    all_scores.to_csv(ANALYSIS / "data" / "intermediate_scores.csv", index=False)
    return all_scores


def write_prediction_metadata(
    output: Path,
    cohort: Path,
    checkpoint: Path,
    table: pd.DataFrame,
    context: str,
    variant: str,
) -> None:
    metadata = {
        "analysis": "Supplementary Figure S1 intermediate-site inference",
        "context": context,
        "variant": variant,
        "graph_version": "graph_v2",
        "rows": int(len(table)),
        "mean_score": float(table["score"].mean()),
        "cohort": str(cohort.relative_to(ANALYSIS)),
        "cohort_sha256": sha256(cohort),
        "checkpoint": str(checkpoint.relative_to(REPRO)),
        "checkpoint_sha256": sha256(checkpoint),
        "output": str(output.relative_to(ANALYSIS)),
        "output_sha256": sha256(output),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "status": "PASS",
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def aggregate(intermediate: pd.DataFrame) -> dict:
    score_rows = []
    threshold_rows = []
    for context in CONTEXTS:
        for variant in VARIANTS:
            inter = intermediate[
                (intermediate["tissue"] == context)
                & (intermediate["variant"] == variant)
            ]
            canonical = pd.read_csv(
                REPRO
                / "checkpoints"
                / f"{variant}_{context}"
                / "test_predictions.csv"
            )
            probability = canonical["prob"].to_numpy(float)
            labels = canonical["label_from_loader"].to_numpy(int)
            negative = probability[labels == 0]
            positive_15 = probability[labels == 1]
            bins = [
                inter.loc[
                    (inter["editing_level"] >= lower)
                    & (inter["editing_level"] < upper),
                    "score",
                ].to_numpy(float)
                for lower, upper in ((1, 5), (5, 10), (10, 15))
            ]
            score_rows.append(
                {
                    "tissue": context,
                    "variant": variant,
                    "m_lt1": float(negative.mean()),
                    "m_1_5": float(bins[0].mean()),
                    "m_5_10": float(bins[1].mean()),
                    "m_10_15": float(bins[2].mean()),
                    "m_ge15": float(positive_15.mean()),
                    "corr_inter": float(
                        inter[["editing_level", "score"]].corr().iloc[0, 1]
                    ),
                    "n_lt1": len(negative),
                    "n_1_5": len(bins[0]),
                    "n_5_10": len(bins[1]),
                    "n_10_15": len(bins[2]),
                    "n_ge15": len(positive_15),
                }
            )
            cuts = {}
            for cutoff in (5, 10, 15):
                if cutoff == 15:
                    positive = positive_15
                else:
                    positive = np.concatenate(
                        [
                            inter.loc[
                                inter["editing_level"] >= cutoff, "score"
                            ].to_numpy(float),
                            positive_15,
                        ]
                    )
                y = np.concatenate(
                    [np.zeros(len(negative), dtype=int), np.ones(len(positive), dtype=int)]
                )
                scores = np.concatenate([negative, positive])
                cuts[f"cut{cutoff}"] = float(roc_auc_score(y, scores))
            threshold_rows.append(
                {
                    "tissue": context,
                    "variant": variant,
                    **cuts,
                    "n_negative_lt1": len(negative),
                    "n_positive_cut5": int(
                        len(positive_15) + (inter["editing_level"] >= 5).sum()
                    ),
                    "n_positive_cut10": int(
                        len(positive_15) + (inter["editing_level"] >= 10).sum()
                    ),
                    "n_positive_cut15": len(positive_15),
                }
            )
    score_table = pd.DataFrame(score_rows)
    threshold_table = pd.DataFrame(threshold_rows)
    score_table.to_csv(ANALYSIS / "data" / "run1_score_by_bin.csv", index=False)
    threshold_table.to_csv(ANALYSIS / "data" / "run2_threshold_auroc.csv", index=False)

    return {
        "score_rows": len(score_table),
        "threshold_rows": len(threshold_table),
        "score_table_sha256": sha256(ANALYSIS / "data" / "run1_score_by_bin.csv"),
        "threshold_table_sha256": sha256(ANALYSIS / "data" / "run2_threshold_auroc.csv"),
    }


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.num_threads)
    device = resolve_device(args.device)
    print(
        f"[environment] python={sys.version.split()[0]} torch={torch.__version__} "
        f"device={device}",
        flush=True,
    )
    fidelity = validate_canonical_inference(args, device)
    if args.validate_only:
        print("PASS: canonical inference fidelity", flush=True)
        return
    scores = score_complete_cohorts(args, device)
    aggregation = aggregate(scores)
    metadata = {
        "analysis": "Supplementary Figure S1 editing-level continuum and threshold relaxation",
        "device": str(device),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "intermediate_rows": int(len(scores) // 2),
        "model_site_scores": int(len(scores)),
        "canonical_inference_fidelity": fidelity,
        "aggregation": aggregation,
        "status": "PASS",
    }
    (ANALYSIS / "data" / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print("PASS: inference and aggregation complete", flush=True)


if __name__ == "__main__":
    main()
