#!/usr/bin/env python
"""
Automated training + evaluation runner for the **two models used in the paper**:
- baseline (GAT with untyped edges)
- bio-aware (typed edges + enriched node features + optional global attention)

What it does
1) Creates a timestamped workspace copy of the current repo (and datasets/).
2) Smoke test (tiny subset) to verify imports/paths.
3) Trains/evaluates requested variants for every train/valid pair under datasets/**/combine_*.
4) Saves:
   - Best checkpoint per (variant, split) as checkpoints/<tissue>/<variant>/<split>/best.pth
   - Per-sample predictions as predictions/<tissue>/<variant>/<split>.jsonl
   - PR/ROC arrays as predictions/<tissue>/<variant>/<split>/curves.npz
   - A Markdown summary table: comprehensive_evaluation.md

Model selection (paper-consistent)
- After each epoch, evaluate on validation set and search over 33 thresholds in [0.1, 0.9].
- Choose the epoch whose best-threshold F1 is maximal; save that checkpoint.
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
from torch import nn
from torch_geometric.loader import DataLoader


# =========================
# Configurable paths
# =========================
SRC_ROOT = Path(__file__).resolve().parent  # current repo root

# Root of datasets that include combine_* subdirectories
DATASETS_SRC_ROOT = SRC_ROOT / "datasets"  # contains Combined/, Liver/, Brain_Cerebellum/, etc.

# Training hyperparams for overnight runs
DEFAULT_EPOCHS = 1000
DEFAULT_BATCH = 256


# =========================
# Utilities
# =========================
def copy_workspace(dst_root: Path):
    if dst_root.exists():
        raise RuntimeError(f"Destination {dst_root} already exists. Remove or pick a new timestamp.")
    shutil.copytree(SRC_ROOT, dst_root, dirs_exist_ok=False)

    ds_src = DATASETS_SRC_ROOT
    ds_dst = dst_root / DATASETS_SRC_ROOT.relative_to(SRC_ROOT)
    if ds_src.exists() and not ds_dst.exists():
        shutil.copytree(ds_src, ds_dst)

    print(f"[copy] Project copied to {dst_root}")
    return dst_root


def ensure_exists(path: Path, desc: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")


def assert_combine_tree(root: Path):
    combine_dirs = list(root.rglob("combine_*"))
    if not combine_dirs:
        raise FileNotFoundError(f"No combine_* directories found under {root}. Ensure datasets are copied.")
    print(f"[check] Found {len(combine_dirs)} combine_* directories under {root}")


def subset_jsonl(src: Path, dst: Path, n: int):
    lines = src.read_text().splitlines()
    dst.write_text("\n".join(lines[:n]))
    return dst


def metrics_from_probs(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds)
    spec = recall_score(labels, preds, pos_label=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "precision": prec, "recall": rec, "specificity": spec, "auc": auc, "threshold": threshold}


def best_threshold_metrics(labels: np.ndarray, probs: np.ndarray, grid=None) -> Dict[str, float]:
    # 33 thresholds, linearly spaced in [0.1, 0.9] (paper-consistent)
    if grid is None:
        grid = np.linspace(0.1, 0.9, 33)
    best = None
    for t in grid:
        m = metrics_from_probs(labels, probs, threshold=t)
        if best is None or m["f1"] > best["f1"]:
            best = m
    return best


def write_preds(path: Path, probs: np.ndarray, labels: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"prob": float(p), "label": int(l)} for p, l in zip(probs, labels)]
    path.write_text("\n".join(json.dumps(row) for row in data))


def save_pr_roc_arrays(out_dir: Path, probs: np.ndarray, labels: np.ndarray, thresholds=None):
    """Save arrays for PR/ROC plotting."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.array(labels)
    probs = np.array(probs)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    precision = []
    recall = []
    tpr = []
    fpr = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        tpr.append(rec)
        fpr.append(fp / (fp + tn + 1e-9))
        precision.append(prec)
        recall.append(rec)

    np.savez_compressed(
        out_dir / "curves.npz",
        thresholds=thresholds,
        precision=np.array(precision),
        recall=np.array(recall),
        fpr=np.array(fpr),
        tpr=np.array(tpr),
        probs=probs,
        labels=labels,
    )


# =========================
# Training helpers per variant
# =========================
def run_baseline(
    train_path: Path,
    val_path: Path,
    workdir: Path,
    epochs: int = DEFAULT_EPOCHS,
    save_dir: Path = None,
) -> Tuple[Dict, np.ndarray, np.ndarray, int, Path]:
    """
    Baseline model (untyped edges).
    Expects the repo to provide:
      - gnnadar_verb_compact.load_jsonl_to_graphs(path)
      - gnnadar_verb_compact.RNAEditingGNN(...)
    """
    sys.path.insert(0, str(workdir))
    import gnnadar_verb_compact as base  

    train_graphs = base.load_jsonl_to_graphs(str(train_path))
    val_graphs = base.load_jsonl_to_graphs(str(val_path))
    train_loader = DataLoader(train_graphs, batch_size=DEFAULT_BATCH, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=DEFAULT_BATCH, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base.RNAEditingGNN(input_dim=8, hidden_dim=32, output_dim=1, num_heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    best = {"f1": -1.0, "epoch": 0, "ckpt": None, "metrics": None, "probs": None, "labels": None}

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out, _ = model(batch)
            loss = loss_fn(out.squeeze(), batch.y.squeeze())
            loss.backward()
            optimizer.step()

        # eval
        model.eval()
        y_prob, y_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out, _ = model(batch)
                probs = out.squeeze().detach().cpu().numpy()
                y_prob.extend(probs.tolist() if probs.ndim else [float(probs)])
                y_true.extend(batch.y.cpu().numpy().tolist())

        y_prob = np.array(y_prob)
        y_true = np.array(y_true)
        m = best_threshold_metrics(y_true, y_prob)

        if m["f1"] > best["f1"] + 1e-5:
            best.update({"f1": m["f1"], "epoch": ep, "metrics": m, "probs": y_prob, "labels": y_true})
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = save_dir / "best.pth"
                torch.save({"epoch": ep, "state_dict": model.state_dict()}, ckpt_path)
                best["ckpt"] = ckpt_path

    return best["metrics"], best["probs"], best["labels"], best["epoch"], best["ckpt"]


def run_bioaware(
    train_path: Path,
    val_path: Path,
    workdir: Path,
    use_gps: bool = False,
    epochs: int = DEFAULT_EPOCHS,
    save_dir: Path = None,
) -> Tuple[Dict, np.ndarray, np.ndarray, int, Path]:
    """
    Bio-aware model (typed edges + enriched node features + optional global attention).
    Expects the repo to provide:
      - models.bioaware_gnn.BioAwareGNN
      - models.bioaware_gnn.load_graphs(path, use_neighbors=True, use_geometry=True, plfold_dir=None)
    """
    sys.path.insert(0, str(workdir))
    from models.bioaware_gnn import BioAwareGNN, load_graphs  

    train_graphs = load_graphs(str(train_path), use_neighbors=True, use_geometry=True, plfold_dir=None)
    val_graphs = load_graphs(str(val_path), use_neighbors=True, use_geometry=True, plfold_dir=None)
    train_loader = DataLoader(train_graphs, batch_size=DEFAULT_BATCH, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=DEFAULT_BATCH, shuffle=False)

    in_dim = 22  # bio-aware node features (paper)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = "bioaware_gps" if use_gps else "bioaware"
    print(f"[{tag}] build model: in_dim={in_dim}", flush=True)

    model = BioAwareGNN(
        in_dim=in_dim,
        hidden=96,
        heads=4,
        layers=3,
        edge_emb_dim=6,
        edge_scalar_dim=12,
        dropout=0.1,
        seq_branch_dim=128,
        use_global_attn=use_gps,
        global_attn_heads=4,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best = {"f1": -1.0, "epoch": 0, "ckpt": None, "metrics": None, "probs": None, "labels": None}
    bad = 0
    patience = epochs  # run full length for comparability

    for ep in range(1, epochs + 1):
        model.train()
        print(f"[{tag}] {train_path.stem}->{val_path.stem} ep {ep:03d} train_start", flush=True)
        for bi, batch in enumerate(train_loader, 1):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if bi % 20 == 0 or bi == 1:
                print(f"[{tag}] {train_path.stem}->{val_path.stem} ep {ep:03d} batch {bi} loss={loss.item():.4f}", flush=True)

        # eval
        y_prob, y_true = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                y_prob.extend(probs.tolist() if probs.ndim else [float(probs)])
                y_true.extend(batch.y.cpu().numpy().tolist())

        y_prob = np.array(y_prob)
        y_true = np.array(y_true)
        m = best_threshold_metrics(y_true, y_prob)
        print(
            f"[{tag}] {train_path.stem}->{val_path.stem} ep {ep:03d} "
            f"f1={m['f1']:.4f} acc={m['acc']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} "
            f"spec={m['specificity']:.4f} thr={m['threshold']:.2f}",
            flush=True,
        )

        if m["f1"] > best["f1"] + 1e-5:
            best.update({"f1": m["f1"], "epoch": ep, "metrics": m, "probs": y_prob, "labels": y_true})
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = save_dir / "best.pth"
                torch.save({"epoch": ep, "state_dict": model.state_dict()}, ckpt_path)
                best["ckpt"] = ckpt_path
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    return best["metrics"], best["probs"], best["labels"], best["epoch"], best["ckpt"]


# =========================
# Orchestration
# =========================
@dataclass
class VariantResult:
    name: str
    split: str
    acc: float
    f1: float
    precision: float
    recall: float
    specificity: float
    auc: float
    preds_path: Path
    best_epoch: int
    ckpt_path: Path


def bold_best(table: List[VariantResult]) -> str:
    metrics = ["acc", "f1", "precision", "recall", "specificity", "auc"]
    best_vals = {m: max(getattr(r, m) for r in table if not np.isnan(getattr(r, m))) for m in metrics}
    lines = [
        "| Model | Split | ACC | F1 | Precision | Recall | Specificity | AUROC | Best Epoch |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in table:
        cells = []
        for m in metrics:
            val = getattr(r, m)
            sval = f"{val:.4f}"
            if abs(val - best_vals[m]) < 1e-9:
                sval = f"**{sval}**"
            cells.append(sval)
        cells.append(str(r.best_epoch))
        lines.append(f"| {r.name} | {r.split} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def discover_pairs(datasets_root: Path) -> List[Tuple[Path, Path, str, str]]:
    """Find all train/val pairs under datasets_root/**/combine_*/ with *_train.jsonl and *_valid.jsonl."""
    pairs: List[Tuple[Path, Path, str, str]] = []

    def combine_sort_key(p: Path):
        name = p.name
        parts = name.split("_")
        if len(parts) >= 3 and parts[0] == "combine":
            try:
                return (int(parts[1]), int(parts[2]), str(p))
            except ValueError:
                return (9999, 9999, str(p))
        return (9999, 9999, str(p))

    combine_dirs = sorted(datasets_root.rglob("combine_*"), key=combine_sort_key)
    for combine_dir in combine_dirs:
        train_files = sorted(combine_dir.glob("*_train.jsonl"))
        val_files = sorted(combine_dir.glob("*_valid.jsonl"))
        if not train_files or not val_files:
            continue
        for tr in train_files:
            train_name = tr.stem.replace("_train", "")
            for va in val_files:
                val_name = va.stem.replace("_valid", "")
                split_label = f"{train_name}->{val_name}"
                pairs.append((tr, va, combine_dir.parent.name, combine_dir.name + "/" + split_label))
    return pairs


def smoke_test(workdir: Path):
    print("[smoke] starting")
    pairs = discover_pairs(workdir / DATASETS_SRC_ROOT.relative_to(SRC_ROOT))
    if not pairs:
        raise RuntimeError("Smoke test failed: no train/valid pairs found.")

    train_path, val_path, _, _ = pairs[0]
    ensure_exists(train_path, "smoke train JSONL")
    ensure_exists(val_path, "smoke val JSONL")

    tmp_train = workdir / "smoke_train.jsonl"
    tmp_val = workdir / "smoke_val.jsonl"
    subset_jsonl(train_path, tmp_train, 20)
    subset_jsonl(val_path, tmp_val, 8)

    smoke_variants = [
        ("baseline", lambda: run_baseline(tmp_train, tmp_val, workdir, epochs=1)),
        ("bioaware", lambda: run_bioaware(tmp_train, tmp_val, workdir, use_gps=False, epochs=1)),
    ]
    for name, fn in smoke_variants:
        m, _, _, _, _ = fn()
        print(f"[smoke] {name}: F1={m.get('f1', float('nan')):.4f}")

    print("[smoke] ok")


def clean_variant_dirs(workdir: Path, variant_names: List[str]):
    for root_name in ["checkpoints", "predictions"]:
        root = workdir / root_name
        if not root.exists():
            continue
        for v in variant_names:
            for path in root.rglob(f"*{v}*"):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.is_file():
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", default=None, help="Optional timestamp directory name")
    ap.add_argument(
        "--variants",
        default="baseline,bioaware",
        help="Comma-separated list of variants to run: baseline,bioaware,bioaware_gps",
    )
    ap.add_argument("--skip_variants", default="", help="Comma-separated list of variants to skip")
    ap.add_argument("--clean_variants", default="", help="Comma-separated list of variants to delete old preds/ckpts for in the copied workspace before running")
    ap.add_argument("--start_after", default="", help="Skip splits lexicographically <= this marker (e.g., 'combine_2_4/BrainCerebellum->MuscleSkeletal')")
    ap.add_argument("--resume_existing", action="store_true", help="Skip a split if predictions already exist for that variant/split in the workspace")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    ts = args.timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_root = SRC_ROOT.parent / f"overnight_eval_{ts}"
    copy_workspace(dst_root)
    assert_combine_tree(dst_root / DATASETS_SRC_ROOT.relative_to(SRC_ROOT))

    os.chdir(dst_root)

    clean_targets = [v.strip() for v in args.clean_variants.split(",") if v.strip()]
    if clean_targets:
        clean_variant_dirs(dst_root, clean_targets)

    smoke_test(dst_root)

    results: List[VariantResult] = []
    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    skips = {v.strip() for v in args.skip_variants.split(",") if v.strip()}

    pairs = discover_pairs(dst_root / DATASETS_SRC_ROOT.relative_to(SRC_ROOT))
    if not pairs:
        raise RuntimeError("No train/val pairs found under datasets/ for evaluation.")
    start_after = args.start_after.strip()

    for variant_name in requested:
        if variant_name in skips:
            continue
        for train_p, val_p, tissue, split_label in pairs:
            if start_after and split_label <= start_after:
                continue

            preds_path = dst_root / "predictions" / tissue / variant_name / f"{split_label.replace('->','_')}.jsonl"
            if args.resume_existing and preds_path.exists():
                print(f"[skip resume] {variant_name} {split_label}: preds already exist at {preds_path}")
                continue

            ckpt_dir = dst_root / "checkpoints" / tissue / variant_name / split_label.replace("->", "_")
            print(f"[run] {variant_name} | split={split_label} | train={train_p} | val={val_p} | ckpt_dir={ckpt_dir}")

            try:
                if variant_name == "baseline":
                    metrics, probs, labels, best_epoch, ckpt_path = run_baseline(train_p, val_p, dst_root, epochs=DEFAULT_EPOCHS, save_dir=ckpt_dir)
                elif variant_name == "bioaware":
                    metrics, probs, labels, best_epoch, ckpt_path = run_bioaware(train_p, val_p, dst_root, use_gps=False, epochs=DEFAULT_EPOCHS, save_dir=ckpt_dir)
                elif variant_name == "bioaware_gps":
                    metrics, probs, labels, best_epoch, ckpt_path = run_bioaware(train_p, val_p, dst_root, use_gps=True, epochs=DEFAULT_EPOCHS, save_dir=ckpt_dir)
                else:
                    raise ValueError(f"Unknown variant: {variant_name}")
            except FileNotFoundError:
                print(f"[skip] {variant_name} {split_label}: missing data ({val_p})")
                continue
            except Exception as e:
                print(f"[error] {variant_name} {split_label}: {e}")
                continue

            write_preds(preds_path, probs, labels)
            save_pr_roc_arrays(preds_path.parent / split_label.replace("->", "_"), probs, labels)

            results.append(
                VariantResult(
                    name=variant_name,
                    split=split_label,
                    acc=metrics["acc"],
                    f1=metrics["f1"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    specificity=metrics["specificity"],
                    auc=metrics["auc"],
                    preds_path=preds_path,
                    best_epoch=best_epoch,
                    ckpt_path=ckpt_path if ckpt_path else ckpt_dir / "best.pth",
                )
            )
            print(f"[done] {variant_name} {split_label}: F1={metrics['f1']:.4f} (best_epoch={best_epoch}) | preds={preds_path} | ckpt={ckpt_path}")

    md = bold_best(results)
    report = textwrap.dedent(
        f"""        # Comprehensive Evaluation

        Workspace: `{dst_root}`

        {md}
        """
    )
    out_md = dst_root / "comprehensive_evaluation.md"
    out_md.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
