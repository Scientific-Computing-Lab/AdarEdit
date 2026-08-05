#!/usr/bin/env python3
"""Create the coding-target AUROC panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent

GENES = ("AJUBA", "BLCAP", "FLNA", "TTYH2", "NEIL1", "GRIA2", "GRIA3")
TISSUES = ("Brain", "Combined", "Liver")
FILE_TISSUE = {
    "Brain": "BrainCerebellum",
    "Combined": "Combined",
    "Liver": "Liver",
}
MODEL_STYLE = {
    "baseline": ("Per-tissue Baseline", "#F2A79D"),
    "bioaware": ("Per-tissue Bio-aware", "#B3A7D6"),
    "joint_baseline": ("Joint Baseline", "#E68A77"),
    "joint": ("Joint Bio-aware", "#7564A8"),
}
JOINT_MODELS = ("joint", "joint_baseline")
PER_TISSUE_MODELS = ("bioaware", "baseline")
ALL_MODELS = ("baseline", "joint_baseline", "bioaware", "joint")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_summary(path: Path):
    frame = pd.read_csv(path)
    frame["auroc"] = pd.to_numeric(frame["auroc"], errors="coerce")
    return frame


def get_value(frame, variant, gene, tissue):
    row = frame[
        (frame["variant"] == variant)
        & (frame["gene"] == gene)
        & (frame["tissue"] == tissue)
    ]
    if len(row) != 1:
        raise ValueError(f"expected one row for {variant}/{gene}/{tissue}")
    return row.iloc[0]


def draw_auroc_axis(ax, frame, tissue, variants, annotate_size=8):
    x = np.arange(len(GENES), dtype=float)
    width = 0.76 / len(variants)
    for variant_index, variant in enumerate(variants):
        label, color = MODEL_STYLE[variant]
        offset = (variant_index - (len(variants) - 1) / 2) * width
        for gene_index, gene in enumerate(GENES):
            row = get_value(frame, variant, gene, tissue)
            value = row["auroc"]
            if np.isnan(value):
                continue
            ax.bar(
                gene_index + offset,
                value,
                width=width,
                color=color,
                edgecolor="#404040",
                linewidth=0.45,
                label=label if gene_index == 0 else None,
                zorder=2,
            )
            ax.text(
                gene_index + offset,
                value + 0.018,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=annotate_size,
            )
    for gene_index, gene in enumerate(GENES):
        if all(
            np.isnan(get_value(frame, variant, gene, tissue)["auroc"])
            for variant in variants
        ):
            ax.text(
                gene_index,
                0.07,
                "n/a",
                color="#999999",
                style="italic",
                ha="center",
            )

    count_rows = [
        get_value(frame, variants[0], gene, tissue) for gene in GENES
    ]
    labels = [
        f"{gene}\n(n={int(row['n_not_edited'])},{int(row['n_edited'])})"
        for gene, row in zip(GENES, count_rows)
    ]
    ax.axhline(
        0.5, color="#999999", linestyle=(0, (4, 3)), linewidth=1.0, zorder=0
    )
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(f"{tissue}\nAUROC", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.7, zorder=0)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")


def auroc_figure(frame, variants, output_stem: Path):
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 10.5))
    for axis, tissue in zip(axes, TISSUES):
        draw_auroc_axis(
            axis,
            frame,
            tissue,
            variants,
            annotate_size=7 if len(variants) == 4 else 9,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(variants),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    save(figure, output_stem)


def save(figure, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"[figure] {stem.with_suffix('.png')}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path, default=PACKAGE / "results" / "auroc_summary.csv"
    )
    parser.add_argument(
        "--output-root", type=Path, default=PACKAGE / "figures"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame = load_summary(args.summary)
    auroc_figure(
        frame,
        JOINT_MODELS,
        args.output_root / "coding_target_auroc_joint",
    )
    auroc_figure(
        frame,
        PER_TISSUE_MODELS,
        args.output_root / "coding_target_auroc_per_tissue",
    )
    auroc_figure(
        frame,
        ALL_MODELS,
        args.output_root / "coding_target_auroc_all_models",
    )


if __name__ == "__main__":
    main()
