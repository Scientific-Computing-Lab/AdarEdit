#!/usr/bin/env python3
"""Render the Liver component-ablation figure."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)


def main() -> None:
    payload = json.loads((BASE / "ablation_summary.json").read_text())
    rows = sorted(
        (row for row in payload["rows"] if row["tag"] != "full"),
        key=lambda row: row["delta_auroc"],
    )
    labels = [row["component"] for row in rows]
    delta_f1 = np.asarray([row["delta_f1"] for row in rows])
    delta_auroc = np.asarray([row["delta_auroc"] for row in rows])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "figure.dpi": 300,
        }
    )

    figure, axis = plt.subplots(figsize=(120 / 25.4, 66 / 25.4))
    positions = np.arange(len(labels))[::-1]
    height = 0.36
    axis.barh(
        positions + height / 2,
        delta_auroc,
        height=height,
        color="#2b6cb0",
        label=r"$\Delta$AUROC",
        zorder=3,
    )
    axis.barh(
        positions - height / 2,
        delta_f1,
        height=height,
        color="#b0b7c0",
        label=r"$\Delta$F1",
        zorder=3,
    )
    axis.axvline(0, color="#333333", linewidth=0.7, zorder=4)
    axis.axvspan(-0.07, 0, color="#2b6cb0", alpha=0.05, zorder=0)

    for y_value, value in zip(positions, delta_auroc):
        axis.text(
            value + (-0.002 if value < 0 else 0.002),
            y_value + height / 2,
            f"{value:+.3f}",
            va="center",
            ha="right" if value < 0 else "left",
            fontsize=6,
            color="#2b6cb0",
        )

    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_xlabel(
        r"$\Delta$ test performance "
        "(ablation $-$ full bio-aware model)"
    )
    axis.set_xlim(-0.07, 0.025)
    axis.legend(frameon=False, loc="lower left", handlelength=1.1)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    stem = OUT / "component_ablation"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(figure)

    manuscript = REPO / "manuscript" / "figS_ablation.png"
    manuscript.parent.mkdir(exist_ok=True)
    shutil.copyfile(stem.with_suffix(".png"), manuscript)
    print(f"wrote {stem.with_suffix('.pdf')}")
    print(f"wrote {stem.with_suffix('.png')}")
    print(f"wrote {manuscript}")


if __name__ == "__main__":
    main()
