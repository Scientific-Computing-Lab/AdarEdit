#!/usr/bin/env python3
"""Render the attention-interpretability figure and standalone panels."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
DATA = ANALYSIS / "data"
FIGURES = ANALYSIS / "figures"
PANELS = FIGURES / "panels"
COLORS = {
    "GAT baseline": "#E64B35",
    "XGBoost all positions": "#4DBBD5",
    "XGBoost top 20": "#00A087",
    "edited": "#E64B35",
    "not_edited": "#4DBBD5",
    "A": "#00A087",
    "C": "#3C5488",
    "G": "#F39B7F",
    "U": "#E64B35",
    "Paired": "#8491B4",
    "Unpaired": "#00A087",
}


def set_style():
    plt.style.use("default")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.05


def panel_metrics(metrics, output):
    test_results = {
        "GAT baseline": metrics["gat_baseline"]["test"],
        "XGBoost all positions": metrics["xgboost_all_features"]["test"],
        "XGBoost top 20": metrics["xgboost_top20_features"]["test"],
    }
    rows = []
    for model, values in test_results.items():
        for key, label in (
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1-Score"),
        ):
            rows.append({"Model": model, "Metric": label, "Score": values[key]})
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(4.3, 3.0))
    sns.barplot(
        data=frame,
        x="Metric",
        y="Score",
        hue="Model",
        palette=COLORS,
        edgecolor="black",
        linewidth=0.3,
        width=0.72,
        ax=axis,
    )
    for patch in axis.patches:
        if patch.get_height() > 0:
            axis.annotate(
                f"{patch.get_height():.2f}",
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                ha="center",
                va="bottom",
                fontsize=6,
                xytext=(0, 2),
                textcoords="offset points",
            )
    axis.set_ylim(0.65, 0.95)
    axis.set_xlabel("")
    axis.set_ylabel("Score on the complete test split")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    figure.tight_layout()
    figure.savefig(output)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def panel_shap(input_path, title, output):
    with input_path.open("rb") as handle:
        content = pickle.load(handle)
    # SHAP's beeswarm uses random vertical jitter; fix it for pixel-stable plots.
    shap.summary_plot(
        content["shap_values"],
        content["X_display"],
        feature_names=content["feature_names"],
        max_display=10,
        show=False,
        color=plt.get_cmap("coolwarm"),
        plot_size=(4, 4),
        rng=np.random.default_rng(42),
    )
    axis = plt.gca()
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("SHAP value (impact on model output)", fontsize=9)
    axis.tick_params(axis="both", labelsize=9)
    plt.tight_layout()
    plt.savefig(output)
    plt.savefig(output.with_suffix(".pdf"))
    plt.close()


def style_profile_axis(axis, window=50):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_position(("outward", 5))
    axis.spines["bottom"].set_position(("outward", 5))
    axis.axvline(0, color="black", linewidth=0.6, alpha=0.4, zorder=1)
    axis.xaxis.set_major_locator(MultipleLocator(10))
    axis.xaxis.set_minor_locator(MultipleLocator(1))
    axis.grid(
        True,
        which="major",
        axis="x",
        linewidth=0.3,
        color="#E0E0E0",
    )
    axis.set_xlim(-window, window)
    axis.set_xlabel("Position relative to candidate editing site")
    axis.set_ylabel("Mean attention")


def panel_edited_status(node_frame, output):
    frame = node_frame
    aggregate = (
        frame.groupby(["rel_pos", "ground_truth"])["attention_L2"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(4.5, 3.0))
    style_profile_axis(axis)
    for label, display, color, linestyle in (
        (1, "Edited", COLORS["edited"], "-"),
        (0, "Not edited", COLORS["not_edited"], "--"),
    ):
        subset = aggregate[aggregate["ground_truth"] == label]
        sem = subset["sem"].fillna(0.0)
        axis.plot(
            subset["rel_pos"],
            subset["mean"],
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=2,
            markeredgecolor="white",
            markeredgewidth=0.2,
            linewidth=0.8,
            label=display,
            zorder=3,
        )
        axis.fill_between(
            subset["rel_pos"],
            subset["mean"] - sem,
            subset["mean"] + sem,
            color=color,
            alpha=0.1,
            linewidth=0,
            zorder=2,
        )
    axis.legend(frameon=False, loc="upper right")
    axis.set_title("Global attention profile", fontweight="bold", fontsize=10)
    figure.tight_layout()
    figure.savefig(output)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def panel_nucleotide(node_frame, output):
    frame = node_frame[node_frame["base"].isin(["A", "C", "G", "U"])]
    aggregate = (
        frame.groupby(["rel_pos", "base"])["attention_L2"].mean().reset_index()
    )
    figure, axis = plt.subplots(figsize=(4.5, 3.0))
    style_profile_axis(axis)
    for base in ("A", "C", "G", "U"):
        subset = aggregate[aggregate["base"] == base]
        axis.plot(
            subset["rel_pos"],
            subset["attention_L2"],
            color=COLORS[base],
            marker="o",
            markersize=2,
            markeredgecolor="white",
            markeredgewidth=0.2,
            linewidth=0.8,
            label=base,
            zorder=3,
        )
    axis.legend(
        title="Base",
        ncol=4,
        loc="upper right",
        fontsize=6,
        frameon=False,
    )
    axis.set_title("Nucleotide preference", fontweight="bold", fontsize=10)
    figure.tight_layout()
    figure.savefig(output)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def panel_structure(node_frame, output):
    frame = node_frame.copy()
    frame["Structure"] = np.where(frame["paired"] == 1, "Paired", "Unpaired")
    aggregate = (
        frame.groupby(["rel_pos", "Structure"])["attention_L2"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(4.5, 3.0))
    style_profile_axis(axis)
    for structure in ("Paired", "Unpaired"):
        subset = aggregate[aggregate["Structure"] == structure]
        sem = subset["sem"].fillna(0.0)
        axis.plot(
            subset["rel_pos"],
            subset["mean"],
            color=COLORS[structure],
            marker="s",
            markersize=2,
            markeredgecolor="white",
            markeredgewidth=0.2,
            linewidth=0.8,
            label=structure,
            zorder=3,
        )
        axis.fill_between(
            subset["rel_pos"],
            subset["mean"] - sem,
            subset["mean"] + sem,
            color=COLORS[structure],
            alpha=0.1,
            linewidth=0,
            zorder=2,
        )
    axis.legend(frameon=False, loc="upper right")
    axis.set_title("Structural context", fontweight="bold", fontsize=10)
    figure.tight_layout()
    figure.savefig(output)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def label_image(path, letter):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38
        )
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 4), letter, fill="black", font=font)
    return image


def make_row(paths, letters, height):
    images = [label_image(paths[letter], letter) for letter in letters]
    widths = [int(image.width * height / image.height) for image in images]
    canvas = Image.new("RGB", (sum(widths) + 20 * (len(images) - 1), height), "white")
    left = 0
    for image, width in zip(images, widths):
        canvas.paste(image.resize((width, height)), (left, 0))
        left += width + 20
    return canvas


def make_full_composite(paths, output):
    upper = make_row(paths, ("B", "C", "D"), 800)
    lower = make_row(paths, ("E", "F", "G"), 620)
    width = max(upper.width, lower.width)
    canvas = Image.new("RGB", (width, upper.height + lower.height + 20), "white")
    canvas.paste(upper, ((width - upper.width) // 2, 0))
    canvas.paste(lower, ((width - lower.width) // 2, upper.height + 20))
    canvas.save(output)


def main():
    set_style()
    PANELS.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(
        (DATA / "metrics_L2.json").read_text()
    )
    node_frame = pd.read_csv(DATA / "node_level_attention_test_L2.csv")
    paths = {
        "B": PANELS / "panel_B_metrics.png",
        "C": PANELS / "panel_C_shap_all_positions.png",
        "D": PANELS / "panel_D_shap_top20.png",
        "E": PANELS / "panel_E_edited_vs_not.png",
        "F": PANELS / "panel_F_nucleotide.png",
        "G": PANELS / "panel_G_structure.png",
    }
    panel_metrics(metrics, paths["B"])
    panel_shap(
        DATA / "shap_validation_all_positions_L2.pkl",
        "Top 10 SHAP Feature Original Model",
        paths["C"],
    )
    panel_shap(
        DATA / "shap_validation_top20_L2.pkl",
        "Top 10 SHAP Feature Retrained Model",
        paths["D"],
    )
    panel_edited_status(node_frame, paths["E"])
    panel_nucleotide(node_frame, paths["F"])
    panel_structure(node_frame, paths["G"])

    graph_frame = node_frame[["graph_index", "ground_truth"]].drop_duplicates()
    population = {
        "panels_E_to_G": {
            "split": "test",
            "filter": "none; all test sites",
            "graphs": int(len(graph_frame)),
            "edited_graphs": int((graph_frame["ground_truth"] == 1).sum()),
            "not_edited_graphs": int((graph_frame["ground_truth"] == 0).sum()),
            "node_rows": int(len(node_frame)),
        },
    }
    (DATA / "descriptive_panel_population.json").write_text(
        json.dumps(population, indent=2, sort_keys=True) + "\n"
    )

    make_full_composite(
        paths,
        FIGURES / "attention_interpretability.png",
    )
    print("Saved panels under figures/panels")


if __name__ == "__main__":
    main()
