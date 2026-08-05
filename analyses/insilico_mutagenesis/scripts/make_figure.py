#!/usr/bin/env python3
"""Render the mutagenesis panels in the manuscript visual style."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
BASES = ("A", "G", "C", "T")
POSITIONS = (-3, -2, -1, 0, 1, 2, 3)

COLOR_STEM = "#2c7bb6"
COLOR_LOOP = "#d7191c"


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def heatmap_matrix(rows, value):
    values = {
        (int(row["position"]), row["base"]): float(row[value])
        for row in rows
    }
    matrix = np.full((4, 7), np.nan)
    for row_index, base in enumerate(BASES):
        for column_index, position in enumerate(POSITIONS):
            if position != 0:
                matrix[row_index, column_index] = values[(position, base)]
    return matrix


def add_target_label(ax):
    ax.text(
        3.5,
        2,
        "Target\n(A)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="gray",
        rotation=90,
    )


def draw_panel_a(ax, rows):
    matrix = heatmap_matrix(rows, "relative_preference")
    limit = np.nanmax(np.abs(matrix))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt=".2f",
        xticklabels=POSITIONS,
        yticklabels=BASES,
        mask=np.isnan(matrix),
        cbar_kws={"label": "Relative Seq Preference"},
    )
    add_target_label(ax)
    ax.set_title("Motif Sequence Preference", fontweight="bold", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")


def draw_panel_b(ax, rows):
    matrix = heatmap_matrix(rows, "mean_delta")
    limit = np.nanmax(np.abs(matrix))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="PuOr",
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt=".2f",
        xticklabels=POSITIONS,
        yticklabels=BASES,
        mask=np.isnan(matrix),
        cbar_kws={"label": "Paired - Unpaired"},
    )
    add_target_label(ax)
    ax.set_title("Structure Preference", fontweight="bold", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")


def draw_panel_c(ax, rows, standalone=False):
    ordered = sorted(rows, key=lambda row: int(row["position"]))
    x = np.asarray([int(row["position"]) for row in ordered])
    y = np.asarray([float(row["mean_delta"]) for row in ordered])
    sem = np.asarray([float(row["sem_delta"]) for row in ordered])

    ax.plot(x, y, color="#333333", linewidth=2.5, zorder=10)
    ax.fill_between(
        x,
        0,
        y,
        where=y >= 0,
        color=COLOR_STEM,
        alpha=0.30,
        interpolate=True,
        label="Prefers Stem",
    )
    ax.fill_between(
        x,
        0,
        y,
        where=y <= 0,
        color=COLOR_LOOP,
        alpha=0.30,
        interpolate=True,
        label="Prefers Loop",
    )
    ax.fill_between(x, y - sem, y + sem, color="black", alpha=0.10, zorder=5)
    ax.axhline(0, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
    sns.despine(ax=ax, top=True, right=True)
    ax.set_xlim(-40, 40)
    axis_fontsize = 18 if standalone else 14
    y_axis_fontsize = 16 if standalone else 11
    tick_fontsize = 15 if standalone else 11
    legend_fontsize = 14 if standalone else 10
    ax.set_xlabel(
        "Relative Position",
        fontsize=axis_fontsize,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Paired - Unpaired Indicator Effect",
        fontsize=y_axis_fontsize,
        fontweight="bold",
        labelpad=2,
    )
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(loc="upper right", frameon=False, fontsize=legend_fontsize)


def interaction_matrix(rows, position):
    matrix = np.full((4, 4), np.nan)
    for row in rows:
        if int(row["position"]) != position:
            continue
        i = BASES.index(row["self_base"])
        j = BASES.index(row["partner_base"])
        matrix[i, j] = float(row["mean_prediction"])
    return matrix


def draw_panel_d(axes, rows):
    matrices = [interaction_matrix(rows, position) for position in (-1, 0, 1)]
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    vmin, vmax = float(finite.min()), float(finite.max())

    for ax, position, matrix in zip(axes, (-1, 0, 1), matrices):
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdBu_r",
            center=(vmin + vmax) / 2,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            xticklabels=BASES,
            yticklabels=BASES,
            mask=np.isnan(matrix),
            cbar_kws={"label": "Prediction Score"},
        )
        ax.set_title(f"Position {position}", fontweight="bold", fontsize=14)
        ax.set_xlabel(
            "Opposing Base (Partner)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel(
            "Sequence Base (Self)",
            fontsize=11,
            fontweight="bold",
        )


def add_panel_label(ax, label, x=-0.08, y=1.08):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )


def save_individual(a_rows, b_rows, c_rows, d_rows):
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_panel_a(ax, a_rows)
    fig.tight_layout()
    fig.savefig(FIGURES / "panel_A_motif_sequence.png", dpi=600,
                bbox_inches="tight")
    fig.savefig(FIGURES / "panel_A_motif_sequence.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_panel_b(ax, b_rows)
    fig.tight_layout()
    fig.savefig(FIGURES / "panel_B_structure.png", dpi=600,
                bbox_inches="tight")
    fig.savefig(FIGURES / "panel_B_structure.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_panel_c(ax, c_rows, standalone=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "panel_C_structural_impact.png", dpi=600,
                bbox_inches="tight")
    fig.savefig(FIGURES / "panel_C_structural_impact.pdf",
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    draw_panel_d(axes, d_rows)
    fig.tight_layout()
    fig.savefig(FIGURES / "panel_D_interaction_matrix.png", dpi=600,
                bbox_inches="tight")
    fig.savefig(FIGURES / "panel_D_interaction_matrix.pdf",
                bbox_inches="tight")
    plt.close(fig)


def save_full(a_rows, b_rows, c_rows, d_rows):
    fig = plt.figure(figsize=(18, 11))
    outer = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=(0.95, 1.05),
        hspace=0.30,
        wspace=0.28,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[0, 2])
    draw_panel_a(ax_a, a_rows)
    draw_panel_b(ax_b, b_rows)
    draw_panel_c(ax_c, c_rows)
    add_panel_label(ax_a, "A")
    add_panel_label(ax_b, "B")
    add_panel_label(ax_c, "C")

    lower = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=outer[1, :],
        wspace=0.24,
    )
    axes_d = [fig.add_subplot(lower[0, index]) for index in range(3)]
    draw_panel_d(axes_d, d_rows)
    add_panel_label(axes_d[0], "D")

    fig.savefig(
        FIGURES / "insilico_mutagenesis_full.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURES / "insilico_mutagenesis_full.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    a_rows = read_csv(DATA / "panel_A_summary.csv")
    b_rows = read_csv(DATA / "panel_B_summary.csv")
    c_rows = read_csv(DATA / "panel_C_summary.csv")
    d_rows = read_csv(DATA / "panel_D_summary.csv")

    sns.set(style="white")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    save_individual(a_rows, b_rows, c_rows, d_rows)
    save_full(a_rows, b_rows, c_rows, d_rows)
    print(f"Saved figures under {FIGURES}")


if __name__ == "__main__":
    main()
