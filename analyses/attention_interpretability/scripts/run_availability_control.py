#!/usr/bin/env python3
"""Quantify and remove positional-availability confounding in the attention probe."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ANALYSIS = Path(__file__).resolve().parents[1]
INPUT = ANALYSIS / "input"
DATA = ANALYSIS / "data"
MODELS = ANALYSIS / "models"
POSITIONS = list(range(-50, 51))
ATTENTION_FEATURES = [f"pos_{position}" for position in POSITIONS]
AVAILABILITY_FEATURES = [f"available_{position}" for position in POSITIONS]
TOP_N = 20
PROXIMAL_POSITIONS = (-2, -1, 0, 1)
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
EXPECTED_XGBOOST_VERSION = "1.7.5"
if xgb.__version__ != EXPECTED_XGBOOST_VERSION:
    raise RuntimeError(
        "This control must use the same XGBoost version as the reported "
        f"attention probe: expected {EXPECTED_XGBOOST_VERSION}, found "
        f"{xgb.__version__}."
    )
if int(xgb.__version__.split(".")[0]) < 2:
    XGB_PARAMS.pop("device", None)


def parse_jsonl_record(line: str) -> tuple[int, int, int]:
    record = json.loads(line)
    messages = record["messages"]
    user = next(message["content"] for message in messages if message["role"] == "user")
    label_text = next(
        message["content"].strip().lower()
        for message in messages
        if message["role"] == "assistant"
    )
    prefix = "L:"
    middle = ", A:A, R:"
    structure_marker = ", Alu Vienna Structure:"
    if not user.startswith(prefix) or middle not in user or structure_marker not in user:
        raise ValueError("unexpected JSONL user-message format")
    left, remainder = user[len(prefix):].split(middle, 1)
    right, _ = remainder.split(structure_marker, 1)
    if label_text not in {"yes", "no"}:
        raise ValueError(f"unexpected label {label_text!r}")
    return len(left), len(right), int(label_text == "yes")


def load_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    attention_path = DATA / f"attention_{split}_L2.csv"
    jsonl_path = INPUT / f"Combined_{split}.jsonl"
    frame = pd.read_csv(attention_path)
    records = [parse_jsonl_record(line) for line in jsonl_path.read_text().splitlines()]
    if len(frame) != len(records):
        raise RuntimeError(f"{split}: attention/JSONL row mismatch")
    if frame["graph_index"].tolist() != list(range(len(frame))):
        raise RuntimeError(f"{split}: graph_index is not row-aligned")

    availability = np.zeros((len(frame), len(POSITIONS)), dtype=np.int8)
    for row_index, (left_length, right_length, label) in enumerate(records):
        if int(frame.iloc[row_index]["ground_truth"]) != label:
            raise RuntimeError(f"{split}:{row_index}: label mismatch")
        for column_index, position in enumerate(POSITIONS):
            availability[row_index, column_index] = int(
                -left_length <= position <= right_length
            )

    availability_frame = pd.DataFrame(availability, columns=AVAILABILITY_FEATURES)
    unavailable = availability == 0
    attention_matrix = frame[ATTENTION_FEATURES].to_numpy(dtype=float)
    unavailable_nonzero = int(np.count_nonzero(attention_matrix[unavailable]))
    if unavailable_nonzero:
        raise RuntimeError(
            f"{split}: {unavailable_nonzero} unavailable attention entries are nonzero"
        )
    complete = availability.all(axis=1)
    summary = {
        "sites": int(len(frame)),
        "positive_sites": int(frame["ground_truth"].sum()),
        "complete_window_sites": int(complete.sum()),
        "complete_window_positive_sites": int(
            frame.loc[complete, "ground_truth"].sum()
        ),
        "available_but_zero_attention_entries": int(
            np.count_nonzero((availability == 1) & (attention_matrix == 0))
        ),
        "unavailable_nonzero_attention_entries": unavailable_nonzero,
    }
    return frame, availability_frame, summary


def metric_set(labels, predictions, probabilities) -> dict[str, float | int]:
    return {
        "n": int(len(labels)),
        "positive_n": int(np.sum(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def evaluate(model, matrix: np.ndarray, labels: np.ndarray):
    probabilities = model.predict_proba(matrix)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return metric_set(labels, predictions, probabilities), probabilities, predictions


def fit_model(matrix: np.ndarray, labels: np.ndarray) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(matrix, labels)
    return model


def availability_table(
    split: str,
    frame: pd.DataFrame,
    availability: pd.DataFrame,
) -> list[dict[str, float | int | str]]:
    rows = []
    labels = frame["ground_truth"].to_numpy(dtype=int)
    for label in (0, 1):
        selector = labels == label
        for position, feature in zip(POSITIONS, AVAILABILITY_FEATURES):
            available_n = int(availability.loc[selector, feature].sum())
            n = int(selector.sum())
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "position": position,
                    "n": n,
                    "available_n": available_n,
                    "missing_n": n - available_n,
                    "missing_fraction": (n - available_n) / n,
                }
            )
    return rows


def main() -> None:
    np.random.seed(42)
    DATA.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    valid, valid_availability, valid_summary = load_split("valid")
    test, test_availability, test_summary = load_split("test")
    valid_groups = set(valid["duplex_group"].astype(str))
    test_groups = set(test["duplex_group"].astype(str))
    overlap = valid_groups & test_groups
    if overlap:
        raise RuntimeError(f"validation/test duplex leakage: {len(overlap)} groups")

    availability_rows = availability_table("valid", valid, valid_availability)
    availability_rows.extend(availability_table("test", test, test_availability))
    pd.DataFrame(availability_rows).to_csv(
        DATA / "position_availability.csv", index=False
    )

    valid_labels = valid["ground_truth"].to_numpy(dtype=int)
    test_labels = test["ground_truth"].to_numpy(dtype=int)

    availability_model = fit_model(
        valid_availability.to_numpy(dtype=np.int8), valid_labels
    )
    availability_model.save_model(MODELS / "xgboost_availability_only.json")
    availability_metrics = {}
    availability_outputs = {}
    for split, matrix, labels in (
        ("valid", valid_availability.to_numpy(dtype=np.int8), valid_labels),
        ("test", test_availability.to_numpy(dtype=np.int8), test_labels),
    ):
        metrics, probabilities, predictions = evaluate(
            availability_model, matrix, labels
        )
        availability_metrics[split] = metrics
        availability_outputs[split] = (probabilities, predictions)

    availability_shap = shap.TreeExplainer(availability_model).shap_values(
        valid_availability.to_numpy(dtype=np.int8)
    )
    availability_importance = np.abs(availability_shap).mean(axis=0)
    availability_ranking = np.argsort(availability_importance)[::-1]
    availability_top = [
        {
            "position": POSITIONS[index],
            "mean_absolute_shap": float(availability_importance[index]),
        }
        for index in availability_ranking[:20]
    ]

    valid_complete_selector = valid_availability.to_numpy(dtype=bool).all(axis=1)
    test_complete_selector = test_availability.to_numpy(dtype=bool).all(axis=1)
    valid_complete = valid.loc[valid_complete_selector].reset_index(drop=True)
    test_complete = test.loc[test_complete_selector].reset_index(drop=True)
    valid_complete_labels = valid_complete["ground_truth"].to_numpy(dtype=int)
    test_complete_labels = test_complete["ground_truth"].to_numpy(dtype=int)
    valid_attention = valid_complete[ATTENTION_FEATURES].to_numpy(dtype=float)
    test_attention = test_complete[ATTENTION_FEATURES].to_numpy(dtype=float)

    original_attention_model = xgb.XGBClassifier()
    original_attention_model.load_model(
        MODELS / "xgboost_all_positions_L2.json"
    )
    original_on_complete_metrics = {}
    for split, matrix, labels in (
        ("valid", valid_attention, valid_complete_labels),
        ("test", test_attention, test_complete_labels),
    ):
        metrics, _, _ = evaluate(original_attention_model, matrix, labels)
        original_on_complete_metrics[split] = metrics

    attention_model = fit_model(valid_attention, valid_complete_labels)
    attention_model.save_model(MODELS / "xgboost_complete_window_attention.json")
    complete_attention_metrics = {}
    complete_attention_outputs = {}
    for split, matrix, labels in (
        ("valid", valid_attention, valid_complete_labels),
        ("test", test_attention, test_complete_labels),
    ):
        metrics, probabilities, predictions = evaluate(attention_model, matrix, labels)
        complete_attention_metrics[split] = metrics
        complete_attention_outputs[split] = (probabilities, predictions)

    complete_shap = shap.TreeExplainer(attention_model).shap_values(valid_attention)
    complete_importance = np.abs(complete_shap).mean(axis=0)
    complete_ranking = np.argsort(complete_importance)[::-1]
    top_indices = complete_ranking[:TOP_N]
    top_features = [ATTENTION_FEATURES[index] for index in top_indices]
    top_positions = [POSITIONS[index] for index in top_indices]
    with (DATA / "shap_complete_window_all_positions.pkl").open("wb") as handle:
        pickle.dump(
            {
                "shap_values": complete_shap,
                "X_display": valid_attention,
                "feature_names": ATTENTION_FEATURES,
                "split": "validation complete-window subset",
            },
            handle,
        )

    valid_top = valid_complete[top_features].to_numpy(dtype=float)
    test_top = test_complete[top_features].to_numpy(dtype=float)
    top_model = fit_model(valid_top, valid_complete_labels)
    top_model.save_model(MODELS / "xgboost_complete_window_top20.json")
    top_metrics = {}
    for split, matrix, labels in (
        ("valid", valid_top, valid_complete_labels),
        ("test", test_top, test_complete_labels),
    ):
        metrics, _, _ = evaluate(top_model, matrix, labels)
        top_metrics[split] = metrics
    top_shap = shap.TreeExplainer(top_model).shap_values(valid_top)
    with (DATA / "shap_complete_window_top20.pkl").open("wb") as handle:
        pickle.dump(
            {
                "shap_values": top_shap,
                "X_display": valid_top,
                "feature_names": top_features,
                "split": "validation complete-window subset",
            },
            handle,
        )

    gat_complete_metrics = {}
    for split, frame in (("valid", valid_complete), ("test", test_complete)):
        labels = frame["ground_truth"].to_numpy(dtype=int)
        predictions = frame["model_prediction"].to_numpy(dtype=int)
        probabilities = frame["model_probability"].to_numpy(dtype=float)
        gat_complete_metrics[split] = metric_set(labels, predictions, probabilities)

    test_predictions = test[
        ["graph_index", "duplex_group", "ground_truth"]
    ].copy()
    test_predictions["availability_probability"] = availability_outputs["test"][0]
    test_predictions["availability_prediction"] = availability_outputs["test"][1]
    test_predictions.to_csv(DATA / "availability_only_test_predictions.csv", index=False)

    complete_predictions = test_complete[
        ["graph_index", "duplex_group", "ground_truth"]
    ].copy()
    complete_predictions["attention_probability"] = complete_attention_outputs["test"][0]
    complete_predictions["attention_prediction"] = complete_attention_outputs["test"][1]
    complete_predictions.to_csv(
        DATA / "complete_window_test_predictions.csv", index=False
    )

    original_metrics = json.loads((DATA / "metrics_L2.json").read_text())
    proximal_availability = {}
    for split, table, frame in (
        ("valid", valid_availability, valid),
        ("test", test_availability, test),
    ):
        proximal_availability[split] = {
            str(position): {
                "available_n": int(table[f"available_{position}"].sum()),
                "n": int(len(frame)),
                "available_fraction": float(table[f"available_{position}"].mean()),
            }
            for position in PROXIMAL_POSITIONS
        }

    results = {
        "status": "PASS",
        "protocol": {
            "attention_source": "frozen Baseline Combined GAT, last GAT layer",
            "xgboost_fit_split": "validation",
            "shap_ranking_split": "validation",
            "final_evaluation_split": "test",
            "test_used_for_fitting_or_feature_selection": False,
            "position_range": [-50, 50],
            "complete_window_rule": "all 101 relative positions are present",
            "availability_only_features": "one binary indicator per relative position",
            "xgboost_parameters": XGB_PARAMS,
        },
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
            "xgboost": xgb.__version__,
        },
        "split_integrity": {
            "validation_duplexes": len(valid_groups),
            "test_duplexes": len(test_groups),
            "overlapping_duplexes": 0,
        },
        "split_counts": {"valid": valid_summary, "test": test_summary},
        "proximal_position_availability": proximal_availability,
        "original_attention_probe_full_test": original_metrics[
            "xgboost_all_features"
        ],
        "original_attention_probe_complete_window_subset": (
            original_on_complete_metrics
        ),
        "availability_only": {
            "metrics": availability_metrics,
            "top20_positions_by_validation_shap": availability_top,
        },
        "complete_window_attention": {
            "metrics": complete_attention_metrics,
            "top20_features_ranked_on_validation": top_features,
            "top20_positions_ranked_on_validation": top_positions,
            "validation_mean_absolute_shap": {
                ATTENTION_FEATURES[index]: float(complete_importance[index])
                for index in top_indices
            },
        },
        "complete_window_top20": {"metrics": top_metrics},
        "gat_baseline_complete_window_subset": gat_complete_metrics,
    }
    (DATA / "availability_control_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )

    print("Availability-control analysis")
    print("=============================")
    for split, summary in (("valid", valid_summary), ("test", test_summary)):
        print(
            f"{split}: full={summary['sites']}, "
            f"complete_window={summary['complete_window_sites']}"
        )
    for name, metrics in (
        ("availability-only", availability_metrics["test"]),
        ("original attention/full test", original_metrics["xgboost_all_features"]["test"]),
        ("original attention/complete-window test", original_on_complete_metrics["test"]),
        ("attention/complete-window test", complete_attention_metrics["test"]),
        ("top20/complete-window test", top_metrics["test"]),
    ):
        print(
            f"{name}: n={metrics['n']} f1={metrics['f1']:.4f} "
            f"auroc={metrics['auroc']:.4f} auprc={metrics['auprc']:.4f}"
        )
    print("complete-window top 20:", ", ".join(top_features))


if __name__ == "__main__":
    main()
