#!/usr/bin/env python3
"""Create Supplementary Fig. S2 from the frozen GTEx v8 expression table."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "data" / "gtex_v8_adar_expression.csv"

SELECTED = [
    ("Brain_Cerebellum", "Brain Cerebellum"),
    ("Artery_Tibial", "Artery Tibial"),
    ("Liver", "Liver"),
    ("Muscle_Skeletal", "Muscle Skeletal"),
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def main() -> None:
    table = pd.read_csv(INPUT).set_index("tissue")
    missing = [tissue for tissue, _ in SELECTED if tissue not in table.index]
    if missing:
        raise RuntimeError(f"Missing selected GTEx tissues: {missing}")

    tissue_ids = [tissue for tissue, _ in SELECTED]
    labels = [label for _, label in SELECTED]
    adar = table.loc[tissue_ids, "ADAR_median_TPM"].to_numpy(dtype=float)
    adarb1 = table.loc[tissue_ids, "ADARB1_median_TPM"].to_numpy(dtype=float)

    x = np.arange(len(tissue_ids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(86 / 25.4, 70 / 25.4))
    ax.bar(
        x - width / 2,
        adar,
        width,
        color="#2171b5",
        label="ADAR (ADAR1)",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        adarb1,
        width,
        color="#d94801",
        label="ADARB1 (ADAR2)",
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Median TPM (GTEx v8)")
    ax.set_xlim(-0.6, len(tissue_ids) - 0.4)
    ax.set_title("Selected tissues", fontsize=7, pad=4)
    ax.text(
        -0.14,
        1.08,
        "a",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.legend(frameon=False, loc="upper right", fontsize=6)

    output_stem = BASE / "figures" / "gtex_tissues"
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    manuscript = REPO / "manuscript" / "gtex_tissues.png"
    manuscript.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_stem.with_suffix(".png"), manuscript)
    print(f"wrote {output_stem.with_suffix('.png')}")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"wrote {manuscript}")


if __name__ == "__main__":
    main()
