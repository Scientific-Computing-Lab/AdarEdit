#!/usr/bin/env python3
"""Render diagnostic panels for the positional-availability control."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


ANALYSIS = Path(__file__).resolve().parents[1]
DATA = ANALYSIS / "data"
FIGURES = ANALYSIS / "figures"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((DATA / "availability_control_metrics.json").read_text())
    availability = pd.read_csv(DATA / "position_availability.csv")
    with (DATA / "shap_complete_window_all_positions.pkl").open("rb") as handle:
        shap_content = pickle.load(handle)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure = plt.figure(figsize=(16.5, 5.4))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.12, 1.2, 1.38), wspace=0.52)

    axis_a = figure.add_subplot(grid[0, 0])
    test = availability[availability["split"] == "test"]
    for label, color, name in ((0, "#4DBBD5", "Not edited"), (1, "#E64B35", "Edited")):
        block = test[test["label"] == label].sort_values("position")
        axis_a.plot(
            block["position"],
            100 * block["missing_fraction"],
            color=color,
            linewidth=1.8,
            label=name,
        )
    axis_a.axvspan(-2, 1, color="#6A51A3", alpha=0.10, linewidth=0)
    axis_a.set_title("Missing positions in the held-out test set", pad=10)
    axis_a.set_xlabel("Position relative to candidate site")
    axis_a.set_ylabel("Missing sites (%)")
    axis_a.legend(frameon=False)
    axis_a.text(
        -0.16,
        1.08,
        "A",
        transform=axis_a.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )

    axis_b = figure.add_subplot(grid[0, 1])
    test_metrics = [
        (
            "Availability only\n($n$=4,864)",
            metrics["availability_only"]["metrics"]["test"],
        ),
        (
            "Attention, full test\n($n$=4,864)",
            metrics["original_attention_probe_full_test"]["test"],
        ),
        (
            "Original model, complete window\n($n$=4,077)",
            metrics["original_attention_probe_complete_window_subset"]["test"],
        ),
        (
            "Refit, complete window\n($n$=4,077)",
            metrics["complete_window_attention"]["metrics"]["test"],
        ),
    ]
    y_positions = np.arange(len(test_metrics))
    auroc_values = np.asarray([block["auroc"] for _, block in test_metrics])
    auprc_values = np.asarray([block["auprc"] for _, block in test_metrics])
    for y_position, auroc, auprc in zip(y_positions, auroc_values, auprc_values):
        axis_b.plot(
            [auroc, auprc],
            [y_position, y_position],
            color="#B8B8B8",
            linewidth=1.2,
            zorder=1,
        )
    axis_b.scatter(
        auroc_values,
        y_positions,
        s=58,
        color="#4DBBD5",
        label="AUROC",
        zorder=2,
    )
    axis_b.scatter(
        auprc_values,
        y_positions,
        s=58,
        marker="s",
        color="#00A087",
        label="AUPRC",
        zorder=2,
    )
    axis_b.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    axis_b.set_yticks(y_positions, [name for name, _ in test_metrics])
    axis_b.invert_yaxis()
    axis_b.set_ylim(len(test_metrics) - 0.45, -0.45)
    axis_b.set_xlim(0.48, 0.92)
    axis_b.set_xticks(np.arange(0.5, 0.91, 0.1))
    axis_b.set_xlabel("Held-out test score")
    axis_b.set_title("Boundary-availability sensitivity", pad=10)
    axis_b.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    axis_b.set_axisbelow(True)
    axis_b.legend(frameon=False, ncol=2, loc="upper right")
    for y_position, auroc, auprc in zip(y_positions, auroc_values, auprc_values):
        axis_b.text(
            auroc - 0.008,
            y_position - 0.17,
            f"{auroc:.2f}",
            color="#287C91",
            ha="right",
            va="bottom",
            fontsize=8,
        )
        axis_b.text(
            auprc + 0.008,
            y_position + 0.22,
            f"{auprc:.2f}",
            color="#007B68",
            ha="left",
            va="top",
            fontsize=8,
        )
    axis_b.text(
        -0.32,
        1.08,
        "B",
        transform=axis_b.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )

    axis_c = figure.add_subplot(grid[0, 2])
    plt.sca(axis_c)
    np.random.seed(42)
    display_names = []
    for name in shap_content["feature_names"]:
        position = int(name.replace("pos_", ""))
        display_names.append(f"{position:+d}" if position else "0")
    shap.summary_plot(
        shap_content["shap_values"],
        shap_content["X_display"],
        feature_names=display_names,
        max_display=10,
        show=False,
        color_bar=True,
        plot_size=None,
    )
    axis_c.set_title("Complete-window attention features", pad=10)
    axis_c.set_xlabel("SHAP value (impact on model output)")
    axis_c.set_ylabel("Position relative to candidate site")
    axis_c.text(
        -0.16,
        1.08,
        "C",
        transform=axis_c.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )

    figure.savefig(FIGURES / "attention_availability_control.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURES / "attention_availability_control.pdf", bbox_inches="tight")
    plt.close(figure)
    print("PASS: diagnostic figure written")


if __name__ == "__main__":
    main()
