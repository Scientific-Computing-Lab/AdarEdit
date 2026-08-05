#!/usr/bin/env python3
"""Plot joint-model performance with and without the Combined output."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

TISSUES = ["Artery", "Brain", "Liver", "MuscleSkeletal"]
TISSUE_LABELS = ["Artery", "Brain", "Liver", "Muscle skeletal"]
VARIANTS = [
    ("baseline", "Baseline", "#EE8A7F"),
    ("bioaware", "Bio-aware", "#6C5B9E"),
]
METRICS = [
    ("f1", "Test F1"),
    ("auroc", "Test AUROC"),
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def read_table(path: Path):
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["variant"], row["tissue"])
            if key in rows:
                raise AssertionError(f"Duplicate control row: {key}")
            rows[key] = row
    expected = {
        (variant, tissue)
        for variant, _, _ in VARIANTS
        for tissue in TISSUES
    }
    if set(rows) != expected:
        raise AssertionError(
            f"Control table keys differ from expected: {set(rows) ^ expected}"
        )
    return rows


def main() -> None:
    analysis_root = Path(__file__).resolve().parents[1]
    table_path = analysis_root / "results/combined_supervision_control.csv"
    figure_dir = analysis_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = read_table(table_path)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 9.5),
        sharey=True,
        constrained_layout=False,
    )
    y = np.arange(len(TISSUES))[::-1]
    panel_labels = iter("ABCD")

    for row_index, (metric, metric_label) in enumerate(METRICS):
        for column_index, (variant, variant_label, color) in enumerate(VARIANTS):
            axis = axes[row_index, column_index]
            for position, tissue in zip(y, TISSUES):
                record = rows[(variant, tissue)]
                with_combined = float(record[f"with_combined_{metric}"])
                no_combined = float(record[f"no_combined_{metric}"])
                axis.plot(
                    [with_combined, no_combined],
                    [position, position],
                    color=color,
                    linewidth=2.5,
                    alpha=0.82,
                    zorder=1,
                )
                axis.scatter(
                    with_combined,
                    position,
                    s=145,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=2.5,
                    zorder=3,
                )
                axis.scatter(
                    no_combined,
                    position,
                    s=145,
                    color=color,
                    edgecolors=color,
                    linewidths=1.5,
                    zorder=3,
                )

            # Use one broad absolute scale in every panel. A data-adaptive,
            # truncated scale would visually exaggerate changes of only a few
            # thousandths between the two single-seed runs.
            axis.set_xlim(0.80, 0.95)
            axis.set_xticks(np.arange(0.80, 0.951, 0.05))
            axis.set_xlabel("score (held-out test)", fontsize=14)
            axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
            axis.set_axisbelow(True)
            axis.tick_params(axis="both", labelsize=13)
            if row_index == 0:
                axis.set_title(variant_label, fontsize=18, fontweight="bold", pad=12)
            axis.text(
                0.01,
                0.95,
                metric_label,
                transform=axis.transAxes,
                fontsize=15,
                fontweight="bold",
                va="top",
            )
            axis.text(
                -0.10,
                1.04,
                next(panel_labels),
                transform=axis.transAxes,
                fontsize=20,
                fontweight="bold",
                va="top",
            )

    for row_index in range(2):
        axes[row_index, 0].set_yticks(y)
        axes[row_index, 0].set_yticklabels(TISSUE_LABELS, fontsize=14)
        axes[row_index, 1].tick_params(axis="y", labelleft=False)

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#555555",
            markeredgewidth=2.2,
            linestyle="",
            markersize=10,
            label="With Combined output",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#555555",
            linestyle="",
            markersize=10,
            label="Without Combined output",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.09, top=0.90, wspace=0.16, hspace=0.30)

    for extension in ("png", "pdf"):
        output = figure_dir / f"combined_supervision_control.{extension}"
        fig.savefig(output, dpi=250, bbox_inches="tight")
        print(f"wrote {output}")
    manuscript_dir = analysis_root.parents[1] / "manuscript"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    manuscript_output = manuscript_dir / "figS_joint_combined_control.png"
    fig.savefig(manuscript_output, dpi=250, bbox_inches="tight")
    print(f"wrote {manuscript_output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
