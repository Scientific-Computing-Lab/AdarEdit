#!/usr/bin/env python3
"""Fit attention-only XGBoost probes using GAT validation attention.

The Baseline Combined GAT is frozen. XGBoost is fitted only on attention
extracted from the validation split. SHAP ranking and top-20 feature
selection are also confined to validation. The test split is used only for
final evaluation.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import sklearn
import torch
import torch_geometric
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
FEATURE_MIN = -50
FEATURE_MAX = 50
TOP_N = 20
XGB_PARAMS = {
    "max_depth": 10,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "min_child_weight": 3,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 42,
    "n_jobs": 8,
}
if int(xgb.__version__.split(".")[0]) < 2:
    XGB_PARAMS.pop("device", None)


def metric_set(labels, predictions, probabilities):
    return {
        "n": int(len(labels)),
        "positive_n": int(np.sum(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def positional_columns(frame):
    columns = [
        column
        for column in frame.columns
        if column.startswith("pos_")
        and FEATURE_MIN <= int(column.split("_", 1)[1]) <= FEATURE_MAX
    ]
    expected = [f"pos_{position}" for position in range(FEATURE_MIN, FEATURE_MAX + 1)]
    if columns != expected:
        raise ValueError(
            f"Expected {len(expected)} ordered positional features; "
            f"found {len(columns)}"
        )
    return columns


def model_output(model, frame, features):
    matrix = frame[features].fillna(0).to_numpy()
    labels = frame["ground_truth"].to_numpy(dtype=int)
    probabilities = model.predict_proba(matrix)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return metric_set(labels, predictions, probabilities), probabilities, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ANALYSIS / "data")
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS / "data")
    parser.add_argument("--model-dir", type=Path, default=ANALYSIS / "models")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    valid = pd.read_csv(args.data_dir / "attention_valid_L2.csv")
    test = pd.read_csv(args.data_dir / "attention_test_L2.csv")
    features = positional_columns(valid)
    if positional_columns(test) != features:
        raise ValueError("Validation and test positional columns differ")

    valid_groups = set(valid["duplex_group"].astype(str))
    test_groups = set(test["duplex_group"].astype(str))
    overlap = valid_groups & test_groups
    if overlap:
        raise ValueError(
            f"Validation/test duplex leakage detected: {len(overlap)} groups"
        )
    integrity = {
        "status": "PASS",
        "validation_sites": int(len(valid)),
        "test_sites": int(len(test)),
        "validation_duplexes": int(len(valid_groups)),
        "test_duplexes": int(len(test_groups)),
        "overlapping_duplexes": 0,
    }
    (args.out_dir / "split_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )

    log_lines = []

    def log(message=""):
        print(message, flush=True)
        log_lines.append(message)

    log("Attention-only XGBoost probe")
    log("============================")
    log(f"XGBoost version: {xgb.__version__}")
    log(f"Features: {len(features)} positions ({FEATURE_MIN}..{FEATURE_MAX})")
    log(
        f"XGBoost fit/SHAP split: validation, n={len(valid)}, "
        f"duplexes={len(valid_groups)}"
    )
    log(
        f"Final evaluation split: test, n={len(test)}, "
        f"duplexes={len(test_groups)}"
    )
    log("Validation/test pair-disjoint verification: PASS")

    valid_matrix = valid[features].fillna(0).to_numpy()
    valid_labels = valid["ground_truth"].to_numpy(dtype=int)

    log()
    log("[1] Fit all-position XGBoost on validation")
    model_all = xgb.XGBClassifier(**XGB_PARAMS)
    model_all.fit(valid_matrix, valid_labels)
    all_model_path = args.model_dir / "xgboost_all_positions_L2.json"
    model_all.save_model(all_model_path)
    all_metrics = {}
    all_outputs = {}
    for split, frame in (("valid", valid), ("test", test)):
        result, probabilities, predictions = model_output(model_all, frame, features)
        all_metrics[split] = result
        all_outputs[split] = (probabilities, predictions)
        log(
            f"  {split}: acc={result['accuracy']:.4f} "
            f"prec={result['precision']:.4f} rec={result['recall']:.4f} "
            f"f1={result['f1']:.4f} auroc={result['auroc']:.4f}"
        )

    log()
    log("[2] Rank positions by mean absolute SHAP on validation")
    validation_shap_all = shap.TreeExplainer(model_all).shap_values(valid_matrix)
    mean_absolute = np.abs(validation_shap_all).mean(axis=0)
    ranking = np.argsort(mean_absolute)[::-1]
    top_indices = ranking[:TOP_N]
    top_features = [features[index] for index in top_indices]
    for rank, index in enumerate(top_indices, 1):
        log(
            f"  {rank:2d}. {features[index]} "
            f"(validation mean |SHAP|={mean_absolute[index]:.6f})"
        )

    with (args.out_dir / "shap_validation_all_positions_L2.pkl").open("wb") as handle:
        pickle.dump(
            {
                "shap_values": validation_shap_all,
                "X_display": valid_matrix,
                "feature_names": features,
                "split": "valid",
            },
            handle,
        )
    (args.out_dir / "top20_validation_features_L2.json").write_text(
        json.dumps(
            {
                "selection_split": "valid",
                "top20_features_ranked": top_features,
                "validation_mean_absolute_shap": {
                    features[index]: float(mean_absolute[index])
                    for index in top_indices
                },
            },
            indent=2,
        )
        + "\n"
    )

    log()
    log("[3] Fit top-20 XGBoost on validation")
    model_top = xgb.XGBClassifier(**XGB_PARAMS)
    model_top.fit(valid[top_features].fillna(0).to_numpy(), valid_labels)
    top_model_path = args.model_dir / "xgboost_top20_L2.json"
    model_top.save_model(top_model_path)
    top_metrics = {}
    top_outputs = {}
    for split, frame in (("valid", valid), ("test", test)):
        result, probabilities, predictions = model_output(
            model_top, frame, top_features
        )
        top_metrics[split] = result
        top_outputs[split] = (probabilities, predictions)
        log(
            f"  {split}: acc={result['accuracy']:.4f} "
            f"prec={result['precision']:.4f} rec={result['recall']:.4f} "
            f"f1={result['f1']:.4f} auroc={result['auroc']:.4f}"
        )

    valid_top_matrix = valid[top_features].fillna(0).to_numpy()
    validation_shap_top = shap.TreeExplainer(model_top).shap_values(
        valid_top_matrix
    )
    with (args.out_dir / "shap_validation_top20_L2.pkl").open("wb") as handle:
        pickle.dump(
            {
                "shap_values": validation_shap_top,
                "X_display": valid_top_matrix,
                "feature_names": top_features,
                "split": "valid",
            },
            handle,
        )

    log()
    log("[4] Frozen GAT metrics on the same splits")
    gat_metrics = {}
    for split, frame in (("valid", valid), ("test", test)):
        labels = frame["ground_truth"].to_numpy(dtype=int)
        predictions = frame["model_prediction"].to_numpy(dtype=int)
        probabilities = frame["model_probability"].to_numpy(dtype=float)
        result = metric_set(labels, predictions, probabilities)
        gat_metrics[split] = result
        log(
            f"  {split}: acc={result['accuracy']:.4f} "
            f"prec={result['precision']:.4f} rec={result['recall']:.4f} "
            f"f1={result['f1']:.4f} auroc={result['auroc']:.4f}"
        )

    predictions = test[
        [
            "graph_index",
            "duplex_group",
            "ground_truth",
            "model_probability",
            "model_prediction",
        ]
    ].copy()
    predictions = predictions.rename(
        columns={
            "model_probability": "gat_probability",
            "model_prediction": "gat_prediction",
        }
    )
    predictions["xgboost_all_probability"] = all_outputs["test"][0]
    predictions["xgboost_all_prediction"] = all_outputs["test"][1]
    predictions["xgboost_top20_probability"] = top_outputs["test"][0]
    predictions["xgboost_top20_prediction"] = top_outputs["test"][1]
    predictions.to_csv(args.out_dir / "predictions_test_L2.csv", index=False)

    results = {
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "xgboost": xgb.__version__,
        },
        "protocol": {
            "attention_model": "Baseline Combined GAT, frozen",
            "gat_gradient_training_split": "train",
            "xgboost_fit_split": "valid",
            "shap_feature_selection_split": "valid",
            "final_evaluation_split": "test",
            "test_used_for_xgboost_fitting_or_feature_selection": False,
            "all_models_evaluated_on_identical_test_sites": True,
            "xgboost_threshold": 0.5,
            "feature_range": [FEATURE_MIN, FEATURE_MAX],
            "top_n": TOP_N,
            "xgboost_parameters": XGB_PARAMS,
            "caveat": (
                "The GAT validation split was used for checkpoint "
                "and GAT threshold selection, but not for gradient training."
            ),
        },
        "split_integrity": integrity,
        "top20_features_ranked": top_features,
        "gat_baseline": gat_metrics,
        "xgboost_all_features": all_metrics,
        "xgboost_top20_features": top_metrics,
    }
    metrics_path = args.out_dir / "metrics_L2.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n")
    log()
    log(f"Saved metrics: data/{metrics_path.name}")
    log(
        "Saved models: "
        f"models/{all_model_path.name}, models/{top_model_path.name}"
    )
    (args.out_dir / "attention_probe_L2.log").write_text(
        "\n".join(log_lines) + "\n"
    )


if __name__ == "__main__":
    main()
