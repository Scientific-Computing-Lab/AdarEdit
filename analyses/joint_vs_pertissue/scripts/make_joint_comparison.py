#!/usr/bin/env python3
"""Joint-vs-per-tissue comparison figures, regenerated from data/comparison_data.csv.

For every tissue and encoder family the table holds four held-out-test scores:
the per-tissue model and the single joint (all-tissues) model, each as a Baseline
GAT and a Bio-aware GNN. Two views are produced:

  joint_comparison_slope    dumbbell/slope plot: per-tissue (open) -> joint (filled),
                            one row per encoder family, Test F1 and Test AUROC panels.
  joint_comparison_stacked  grouped bars, the four models side by side per tissue,
                            Test F1 (top) and Test AUROC (bottom, hatched).

Run:  python make_joint_comparison.py
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DATA = BASE / "data" / "comparison_data.csv"
FIG  = BASE / "figures"; FIG.mkdir(exist_ok=True)

TIS   = ["Artery", "Brain", "Combined", "Liver", "MuscleSkeletal"]
TLAB  = ["Artery", "Brain", "Combined", "Liver", "Muscle"]
BIO, BASECOL = "#6C5B9E", "#EE8A7F"                 # encoder family colours
# four-model palette (per-tissue = lighter, joint = saturated)
PTB, PTBio, JB, JBio = "#F19C90", "#B3A7D6", "#C0574A", "#5A4A97"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})

# ---- load: d[tissue][model] = (f1, auroc) ----
d = {t: {} for t in TIS}
for r in csv.DictReader(open(DATA)):
    d[r["tissue"]][r["model"]] = (float(r["f1"]), float(r["auroc"]))


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================ slope / dumbbell ============================
def slope():
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    y = np.arange(len(TIS))[::-1]            # Artery at top
    off = 0.18
    for ax, metric, title in [(axes[0], 0, "Test F1"), (axes[1], 1, "Test AUROC")]:
        for yi, t in zip(y, TIS):
            for sub, per_m, joint_m, col in [
                    (+off, "Per-tissue Bio-aware", "Joint Bio-aware", BIO),
                    (-off, "Per-tissue Baseline", "Joint Baseline", BASECOL)]:
                per = d[t][per_m][metric]; joi = d[t][joint_m][metric]
                ax.plot([per, joi], [yi + sub, yi + sub], color=col, lw=2.2, zorder=1)
                ax.scatter(per, yi + sub, s=130, facecolors="white",
                           edgecolors=col, linewidths=2.2, zorder=3)   # per-tissue = open
                ax.scatter(joi, yi + sub, s=130, color=col, zorder=3)  # joint = filled
        ax.set_title(title, fontsize=17, fontweight="bold", pad=12)
        ax.set_xlabel("score (held-out test)", fontsize=15)
        ax.grid(axis="x", ls="-", lw=0.5, color="#DDDDDD")
        ax.set_axisbelow(True)
    axes[0].set_yticks(y); axes[0].set_yticklabels(TLAB, fontsize=15)
    handles = [Line2D([0], [0], color=BIO, marker="o", lw=2.2, ms=9, label="Bio-aware"),
               Line2D([0], [0], color=BASECOL, marker="o", lw=2.2, ms=9, label="Baseline"),
               Line2D([0], [0], color="#555", marker="o", mfc="white", ls="", ms=9,
                      label="Per-tissue (open)"),
               Line2D([0], [0], color="#555", marker="o", ls="", ms=9, label="Joint (filled)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "joint_comparison_slope")
    print("wrote joint_comparison_slope")


# ============================ grouped bars ============================
def stacked():
    models = [("Per-tissue Baseline", PTB), ("Joint Baseline", JB),
              ("Per-tissue Bio-aware", PTBio), ("Joint Bio-aware", JBio)]
    x = np.arange(len(TIS)); w = 0.20
    fig, (axf, axa) = plt.subplots(2, 1, figsize=(15, 11))
    for ax, metric, title, hatch, lo in [
            (axf, 0, "Test F1", None, 0.79), (axa, 1, "Test AUROC", "////", 0.795)]:
        for k, (m, col) in enumerate(models):
            off = (k - 1.5) * w
            vals = [d[t][m][metric] for t in TIS]
            ax.bar(x + off, vals, w, color=col, edgecolor="white", lw=0.4,
                   hatch=hatch, label=m)
            for xi, v in zip(x, vals):
                ax.text(xi + off, v + 0.001, f"{v:.3f}".lstrip("0"),
                        ha="center", va="bottom", fontsize=9, rotation=90)
        ax.set_ylim(lo, 0.955)
        ax.set_ylabel("score (held-out test)", fontsize=14)
        ax.text(0.008, 0.94, title, transform=ax.transAxes, fontsize=17,
                fontweight="bold", va="top")
        ax.set_xticks(x)
    axf.set_xticklabels([]); axa.set_xticklabels(TLAB, fontsize=14)
    axf.legend(loc="upper center", ncol=4, frameon=False, fontsize=13,
               bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    _save(fig, "joint_comparison_stacked")
    print("wrote joint_comparison_stacked")


if __name__ == "__main__":
    slope()
    stacked()
