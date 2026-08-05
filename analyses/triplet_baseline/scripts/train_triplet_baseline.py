#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

SEED = 42          # fixed so every variant is reproducible run-to-run


REPO = Path(__file__).resolve().parents[3]

BASES = ["A", "C", "G", "T"]
STRUCT_PATTERNS = ["(((", "((.", "(.(", "(..", ".((", ".(.", "..(", "..."]


def paired_char(ch: str) -> str:
    return "." if ch == "." else "("


def triplet_feature_names() -> list[str]:
    return [f"{base}:{pattern}" for base in BASES for pattern in STRUCT_PATTERNS]


def target_triplet_features(sequence: str, structure: str, target_idx_0: int, radius: int) -> np.ndarray:
    sequence = sequence.upper().replace("U", "T")
    feature_index = {name: idx for idx, name in enumerate(triplet_feature_names())}
    vec = np.zeros(len(feature_index) + 4, dtype=float)

    left = max(1, target_idx_0 - radius)
    right = min(len(sequence) - 2, target_idx_0 + radius)
    counted = 0
    paired_center = 0
    for idx in range(left, right + 1):
        base = sequence[idx]
        if base not in BASES:
            continue
        pattern = "".join(paired_char(structure[j]) for j in (idx - 1, idx, idx + 1))
        feature_name = f"{base}:{pattern}"
        vec[feature_index[feature_name]] += 1.0
        counted += 1
        if structure[idx] != ".":
            paired_center += 1

    if counted:
        vec[: len(feature_index)] /= counted
        vec[len(feature_index) + 0] = counted / (2 * radius + 1)
        vec[len(feature_index) + 1] = paired_center / counted

    target_char = structure[target_idx_0]
    vec[len(feature_index) + 2] = float(target_char != ".")
    vec[len(feature_index) + 3] = float(target_idx_0 / max(1, len(sequence) - 1))
    return vec


def build_matrix(df: pd.DataFrame, radius: int) -> tuple[np.ndarray, np.ndarray]:
    features = np.vstack(
        [
            target_triplet_features(
                sequence=row.full_seq,
                structure=row.structure,
                target_idx_0=int(row.target_idx_0),
                radius=radius,
            )
            for row in df.itertuples(index=False)
        ]
    )
    labels = (df["yes_no"] == "yes").astype(int).to_numpy()
    return features, labels


def metric_dict(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    return {
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(recall_score(labels, preds, pos_label=0, zero_division=0)),
        "auc": float(auc),
        "threshold": float(threshold),
    }


def best_threshold(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    best = None
    for thr in np.linspace(0.1, 0.9, 33):
        current = metric_dict(labels, probs, threshold=float(thr))
        if best is None or current["f1"] > best["f1"]:
            best = current
    return best


def make_estimator(model_name: str, c_value: float, gamma: float) -> Pipeline:
    if model_name == "logreg":
        clf = LogisticRegression(C=c_value, max_iter=2000, solver="liblinear",
                                 random_state=SEED)
    elif model_name == "linear_svm":
        clf = LinearSVC(C=c_value, dual="auto", random_state=SEED)
    elif model_name == "rbf_svm":
        # probability=True fits Platt scaling by internal CV; seed it so the
        # reported numbers are reproducible.
        clf = SVC(C=c_value, gamma=gamma, kernel="rbf", probability=True,
                  random_state=SEED)
    else:
        raise ValueError(f"Unsupported model_name={model_name}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def probabilities(estimator: Pipeline, features: np.ndarray) -> np.ndarray:
    clf = estimator.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    decision = estimator.decision_function(features)
    return 1.0 / (1.0 + np.exp(-decision))


def load_split_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"full_seq": "string", "structure": "string", "L": "string", "yes_no": "string"},
    )
    df = df[df["L"].notna()].copy()
    df = df[~df["full_seq"].astype(str).str.contains("nan", case=False, na=False)].copy()
    df = df[df["full_seq"].astype(str).str.len() == df["structure"].astype(str).str.len()].copy()
    df["target_idx_0"] = df["L"].str.len().astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a target-centered Triplet-SVM-style baseline on pair-disjoint splits.")
    parser.add_argument("--split_root", default=str(REPO / "data/human"))
    parser.add_argument("--tissue", required=True)
    parser.add_argument("--model", choices=["logreg", "linear_svm", "rbf_svm"], default="linear_svm")
    parser.add_argument("--radius", type=int, default=25)
    parser.add_argument("--c_values", default="0.1,1,10")
    parser.add_argument("--gamma_values", default="scale,0.1,1")
    parser.add_argument("--out", default=None)
    parser.add_argument("--predictions_out", default=None)
    args = parser.parse_args()

    split_root = Path(args.split_root) / args.tissue
    train_df = load_split_metadata(split_root / "train.metadata.csv")
    valid_df = load_split_metadata(split_root / "valid.metadata.csv")
    test_df = load_split_metadata(split_root / "test.metadata.csv")

    x_train, y_train = build_matrix(train_df, radius=args.radius)
    x_valid, y_valid = build_matrix(valid_df, radius=args.radius)
    x_test, y_test = build_matrix(test_df, radius=args.radius)

    c_values = [float(x) for x in args.c_values.split(",")]
    gamma_values: list[str | float] = [
        (float(x) if x not in {"scale", "auto"} else x) for x in args.gamma_values.split(",")
    ]

    best = None
    for c_value in c_values:
        candidate_gammas = gamma_values if args.model == "rbf_svm" else ["scale"]
        for gamma in candidate_gammas:
            estimator = make_estimator(args.model, c_value=c_value, gamma=gamma if isinstance(gamma, float) else gamma)
            estimator.fit(x_train, y_train)
            valid_probs = probabilities(estimator, x_valid)
            valid_metrics = best_threshold(y_valid, valid_probs)
            print(
                f"[triplet][{args.model}] C={c_value} gamma={gamma} "
                f"valid_f1={valid_metrics['f1']:.4f} valid_auc={valid_metrics['auc']:.4f} "
                f"thr={valid_metrics['threshold']:.3f}",
                flush=True,
            )
            if best is None or valid_metrics["f1"] > best["valid_metrics"]["f1"]:
                best = {
                    "estimator": estimator,
                    "c_value": c_value,
                    "gamma": gamma,
                    "valid_metrics": valid_metrics,
                }

    test_probs = probabilities(best["estimator"], x_test)
    test_metrics = metric_dict(y_test, test_probs, threshold=float(best["valid_metrics"]["threshold"]))

    result = {
        "variant": f"triplet_{args.model}",
        "feature_radius": args.radius,
        "best_hyperparams": {"C": best["c_value"], "gamma": best["gamma"]},
        "valid_metrics": best["valid_metrics"],
        "test_metrics": test_metrics,
        "sizes": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
    }

    out_path = Path(args.out) if args.out else REPO / "analyses/triplet_baseline/results" / f"{args.tissue}_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")

    predictions_out = (
        Path(args.predictions_out)
        if args.predictions_out
        else out_path.with_suffix(".test_predictions.csv")
    )
    pred_df = test_df.copy()
    pred_df["prob"] = test_probs.astype(float)
    pred_df["label_from_loader"] = y_test.astype(int)
    pred_df["pred_label"] = (test_probs >= float(best["valid_metrics"]["threshold"])).astype(int)
    pred_df["threshold"] = float(best["valid_metrics"]["threshold"])
    pred_df["variant"] = result["variant"]
    pred_df.to_csv(predictions_out, index=False)
    print(f"Wrote {predictions_out}")


if __name__ == "__main__":
    main()
