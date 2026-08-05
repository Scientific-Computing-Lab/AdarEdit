#!/usr/bin/env python3
"""Build the training-dynamics supplementary figure from the shipped runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPO = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
CHECKPOINTS = Path(
    os.environ.get("ADAREDIT_DYNAMICS_CHECKPOINTS", REPO / "checkpoints")
)
SMOOTHING_WINDOW = 51

SETTINGS = [
    ("Artery", "Artery Tibial", "#E8743B", False),
    ("Brain", "Brain Cerebellum", "#1B9E77", False),
    ("Liver", "Liver", "#7570B3", False),
    ("MuscleSkeletal", "Muscle Skeletal", "#E7298A", False),
    ("Combined", "Combined", "#2F3B40", False),
    ("Octopus", r"$O.\ bimaculoides$", "#D6A22B", True),
    ("Ptychodera", r"$P.\ flava$", "#17BECF", True),
    ("Strongylocentrotus", r"$S.\ purpuratus$", "#6C8EBF", True),
]
ARCHITECTURES = [("baseline", "Baseline GAT", "--"), ("bioaware", "Bio-aware GNN", "-")]
REQUIRED_HISTORY_COLUMNS = {
    "epoch",
    "train_loss",
    "selection_split",
    "selection_f1",
    "selection_threshold",
    "best_epoch_so_far",
}
GRAPH_INVARIANTS = {
    "baseline_pair_edge_multiplicity_per_direction": 1,
    "stem_feature": "true_contiguous_integer_stem_length",
    "sequential_edge_pair_compatibility": 0.0,
    "t_and_u_share_one_hot_channel": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def load_run(variant: str, setting: str) -> dict:
    run_dir = CHECKPOINTS / f"{variant}_{setting}"
    history_path = run_dir / "history.csv"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "run_config.json"
    for path in (history_path, summary_path, config_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty required input: {path}")

    with history_path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not REQUIRED_HISTORY_COLUMNS.issubset(rows[0]):
        missing = REQUIRED_HISTORY_COLUMNS.difference(rows[0] if rows else {})
        raise RuntimeError(f"{history_path}: missing history columns {sorted(missing)}")

    epoch = np.asarray([int(row["epoch"]) for row in rows], dtype=int)
    loss = np.asarray([float(row["train_loss"]) for row in rows], dtype=float)
    f1 = np.asarray([float(row["selection_f1"]) for row in rows], dtype=float)
    threshold = np.asarray(
        [float(row["selection_threshold"]) for row in rows], dtype=float
    )
    if len(rows) != 1000 or not np.array_equal(epoch, np.arange(1, 1001)):
        raise RuntimeError(
            f"{history_path}: expected exactly one row for every epoch 1--1000"
        )
    if not all(row["selection_split"] == "valid" for row in rows):
        raise RuntimeError(f"{history_path}: non-validation selection row found")
    if not (
        np.isfinite(loss).all()
        and np.isfinite(f1).all()
        and np.isfinite(threshold).all()
    ):
        raise RuntimeError(f"{history_path}: non-finite training value found")

    summary = json.loads(summary_path.read_text())
    config = json.loads(config_path.read_text())
    if summary.get("selection_split") != "valid":
        raise RuntimeError(f"{summary_path}: model was not selected on validation")
    if summary.get("test_used_during_training", False):
        raise RuntimeError(f"{summary_path}: test was used during training")
    if int(summary.get("seed", -1)) != 42:
        raise RuntimeError(f"{summary_path}: expected seed 42")
    if int(summary.get("epochs_requested", -1)) != 1000:
        raise RuntimeError(f"{summary_path}: expected 1,000 requested epochs")
    for key, expected in GRAPH_INVARIANTS.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"{config_path}: {key}={config.get(key)!r}; expected {expected!r}"
            )

    best_index = int(np.argmax(f1))
    best_epoch = int(summary["best_epoch"])
    valid_metrics = summary["valid_selection_metrics"]
    selected_threshold = float(valid_metrics["threshold"])
    if best_epoch != int(epoch[best_index]):
        raise RuntimeError(f"{summary_path}: best epoch differs from history")
    if abs(float(valid_metrics["f1"]) - float(f1[best_index])) > 1e-12:
        raise RuntimeError(f"{summary_path}: best validation F1 differs from history")
    if abs(selected_threshold - float(threshold[best_index])) > 1e-12:
        raise RuntimeError(f"{summary_path}: selected threshold differs from history")
    if (
        abs(float(summary["test_metrics"]["threshold"]) - selected_threshold)
        > 1e-12
    ):
        raise RuntimeError(
            f"{summary_path}: test evaluation did not use validation threshold"
        )

    return {
        "variant": variant,
        "setting": setting,
        "epoch": epoch,
        "loss": loss,
        "f1": f1,
        "threshold": threshold,
        "running_best_f1": np.maximum.accumulate(f1),
        "best_epoch": best_epoch,
        "selected_threshold": selected_threshold,
        "best_valid_f1": float(valid_metrics["f1"]),
        "history_path": display_path(history_path),
        "history_sha256": sha256(history_path),
        "summary_path": display_path(summary_path),
        "summary_sha256": sha256(summary_path),
        "data_sha256": summary["data_sha256"],
    }


def smooth(values: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    if len(values) < window:
        return values
    padding = window // 2
    padded = np.pad(values, padding, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def write_summary(runs: list[dict]) -> dict:
    rows = []
    for run in runs:
        rows.append(
            {
                "architecture": run["variant"],
                "setting": run["setting"],
                "best_epoch": run["best_epoch"],
                "best_valid_f1": run["best_valid_f1"],
                "selected_threshold": run["selected_threshold"],
                "history_path": run["history_path"],
                "history_sha256": run["history_sha256"],
                "summary_path": run["summary_path"],
                "summary_sha256": run["summary_sha256"],
                "data_sha256": run["data_sha256"],
            }
        )
    threshold_summary = {}
    for variant, _, _ in ARCHITECTURES:
        values = np.asarray(
            [
                row["selected_threshold"]
                for row in rows
                if row["architecture"] == variant
            ],
            dtype=float,
        )
        threshold_summary[variant] = {
            "n": int(len(values)),
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "values": values.tolist(),
        }
    payload = {
        "analysis": "training dynamics and validation-selected thresholds",
        "settings": "five human contexts and three within-species models",
        "architectures": ["baseline", "bioaware"],
        "selection_split": "valid",
        "test_used_during_training": False,
        "seed": 42,
        "epochs": 1000,
        "smoothing_window_epochs": SMOOTHING_WINDOW,
        "threshold_summary": threshold_summary,
        "runs": rows,
    }
    (BASE / "training_dynamics_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    columns = [
        "architecture",
        "setting",
        "best_epoch",
        "best_valid_f1",
        "selected_threshold",
        "history_path",
        "history_sha256",
        "summary_path",
        "summary_sha256",
        "data_sha256",
    ]
    with (BASE / "training_dynamics_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["data_sha256"] = json.dumps(
                output["data_sha256"], sort_keys=True, separators=(",", ":")
            )
            writer.writerow(output)
    return payload


def make_figure(runs: list[dict]) -> None:
    indexed = {(run["variant"], run["setting"]): run for run in runs}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "figure.dpi": 300,
        }
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(190 / 25.4, 130 / 25.4),
        gridspec_kw={"height_ratios": [1.0, 1.05]},
    )
    loss_axis, f1_axis, threshold_axis, best_axis, distribution_axis, legend_axis = (
        axes.flat
    )

    for setting, _, colour, _ in SETTINGS:
        for variant, _, linestyle in ARCHITECTURES:
            run = indexed[(variant, setting)]
            loss_axis.plot(
                run["epoch"],
                smooth(run["loss"]),
                color=colour,
                linestyle=linestyle,
                linewidth=0.85,
                alpha=0.85,
            )
            f1_axis.plot(
                run["epoch"],
                smooth(run["f1"]),
                color=colour,
                linestyle=linestyle,
                linewidth=0.85,
                alpha=0.85,
            )
            threshold_axis.plot(
                run["epoch"],
                smooth(run["threshold"]),
                color=colour,
                linestyle=linestyle,
                linewidth=0.8,
                alpha=0.8,
            )
            best_axis.plot(
                run["epoch"],
                run["running_best_f1"],
                color=colour,
                linestyle=linestyle,
                linewidth=0.85,
                alpha=0.85,
            )

    panel_specs = [
        (
            loss_axis,
            "a",
            "Training loss",
            "training loss",
        ),
        (
            f1_axis,
            "b",
            "Validation F1",
            "validation F1",
        ),
        (
            threshold_axis,
            "c",
            "Per-epoch F1-optimal threshold",
            "decision threshold",
        ),
        (
            best_axis,
            "d",
            "Running-best validation F1",
            "best validation F1 so far",
        ),
    ]
    for axis, letter, title, ylabel in panel_specs:
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=9, pad=6)
        axis.text(
            -0.17,
            1.10,
            letter,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=11,
            va="top",
            ha="left",
        )
        axis.grid(linestyle=":", linewidth=0.35, alpha=0.4)
    loss_axis.text(
        0.98,
        0.95,
        f"51-epoch moving average",
        transform=loss_axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#666666",
    )
    threshold_axis.axhline(0.5, linestyle=":", linewidth=0.8, color="#888888")
    threshold_axis.set_ylim(0.0, 0.95)

    thresholds = {
        variant: np.asarray(
            [
                indexed[(variant, setting)]["selected_threshold"]
                for setting, _, _, _ in SETTINGS
            ],
            dtype=float,
        )
        for variant, _, _ in ARCHITECTURES
    }
    colours = {setting: colour for setting, _, colour, _ in SETTINGS}
    rng = np.random.default_rng(1)
    jitter = (rng.random(len(SETTINGS)) - 0.5) * 0.22
    for x_position, (variant, _, _) in enumerate(ARCHITECTURES):
        values = thresholds[variant]
        distribution_axis.boxplot(
            values,
            positions=[x_position],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#2F3B40", "linewidth": 1.4},
            boxprops={
                "facecolor": "#EEEEEE",
                "alpha": 0.7,
                "edgecolor": "#999999",
            },
            whiskerprops={"color": "#999999"},
            capprops={"color": "#999999"},
        )
        for index, (setting, _, _, _) in enumerate(SETTINGS):
            distribution_axis.scatter(
                x_position + jitter[index],
                values[index],
                s=25,
                zorder=4,
                color=colours[setting],
                edgecolor="white",
                linewidth=0.5,
            )
    distribution_axis.axhline(
        0.5, linestyle="--", linewidth=0.9, color="#999999"
    )
    distribution_axis.set_xticks([0, 1])
    distribution_axis.set_xticklabels(["Baseline\nGAT", "Bio-aware\nGNN"])
    distribution_axis.set_ylabel("validation-selected threshold")
    distribution_axis.set_ylim(0.0, 0.65)
    distribution_axis.set_xlim(-0.5, 1.5)
    distribution_axis.set_title("Selected decision thresholds", fontsize=9, pad=6)
    distribution_axis.text(
        -0.17,
        1.10,
        "e",
        transform=distribution_axis.transAxes,
        fontweight="bold",
        fontsize=11,
        va="top",
        ha="left",
    )

    legend_axis.axis("off")
    setting_handles = [
        Line2D([0], [0], color=colour, linewidth=2.2, label=label)
        for _, label, colour, _ in SETTINGS
    ]
    architecture_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
        )
        for _, label, linestyle in ARCHITECTURES
    ]
    first_legend = legend_axis.legend(
        handles=setting_handles,
        loc="upper center",
        ncol=2,
        fontsize=6.7,
        frameon=False,
        title="training / validation setting",
        title_fontsize=7.5,
        bbox_to_anchor=(0.5, 1.02),
        labelspacing=0.35,
        columnspacing=0.8,
        handlelength=2.0,
    )
    legend_axis.add_artist(first_legend)
    legend_axis.legend(
        handles=architecture_handles,
        loc="lower center",
        ncol=1,
        fontsize=7,
        frameon=False,
        title="architecture",
        title_fontsize=7.5,
        bbox_to_anchor=(0.5, 0.02),
    )

    figure.tight_layout()
    stem = BASE / "training_dynamics"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(figure)

    manuscript = REPO / "manuscript" / "training_dynamics.png"
    manuscript.parent.mkdir(exist_ok=True)
    manuscript.write_bytes(stem.with_suffix(".png").read_bytes())
    print(f"wrote {stem.with_suffix('.pdf')}")
    print(f"wrote {stem.with_suffix('.png')}")
    print(f"wrote {manuscript}")


def main() -> None:
    runs = [
        load_run(variant, setting)
        for setting, _, _, _ in SETTINGS
        for variant, _, _ in ARCHITECTURES
    ]
    payload = write_summary(runs)
    make_figure(runs)
    for variant, label, _ in ARCHITECTURES:
        values = payload["threshold_summary"][variant]
        print(
            f"{label}: median={values['median']:.4f}, "
            f"range={values['minimum']:.3f}--{values['maximum']:.3f}"
        )


if __name__ == "__main__":
    main()
