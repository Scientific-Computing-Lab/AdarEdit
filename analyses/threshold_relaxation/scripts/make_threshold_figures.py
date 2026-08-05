#!/usr/bin/env python3
"""Threshold-relaxation figures, regenerated from the pre-aggregated run tables.

Inputs (all in ../data/):
  run0_distribution.csv    per-tissue fraction of sites in each editing-level bin
                           (<1, 1-5, 5-10, 10-15, >=15).
  run1_score_by_bin.csv    per-tissue, per-variant mean predicted score in each
                           editing-level bin (m_lt1 ... m_ge15).
  run2_threshold_auroc.csv per-tissue, per-variant test AUROC when the positive-label
                           cutoff is relaxed to 5 / 10 / 15 %.

Panels:
  (a) stacked editing-level distribution per tissue.
  (b) mean predicted score across the editing-level continuum: bold line = mean
      over tissues, faint lines = individual tissues (baseline vs bioaware).
  (c) AUROC as the positive-label cutoff moves 5 -> 10 -> 15 %, same bold/faint scheme.

Two figures are written to ../figures/:
  threshold_relaxation      = panels (a) + (b) + (c)
  intermediate_site_scores  = panels (b) + (c)

Run:  python make_threshold_figures.py
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DATA = BASE / "data"
FIG  = BASE / "figures"; FIG.mkdir(exist_ok=True)
REPO = BASE.parents[1]
MANUSCRIPT = REPO / "manuscript"

TIS   = ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]
TLAB  = ["Artery", "Brain", "Liver", "Muscle\nskeletal", "Combined"]
VARS  = ["baseline", "bioaware"]
VCOL  = {"baseline": "#EE8A7F", "bioaware": "#5FA98C"}
BINS  = ["<1", "1-5", "5-10", "10-15", "≥15"]          # editing-level bins
STACK = ["#2C7FB8", "#7FB3D8", "#DCE9F4", "#FDBE85", "#E0500A"]
SLAB  = ["<1%", "1–5%", "5–10%", "10–15%", "≥15%"]


def _load(name):
    return list(csv.DictReader(open(DATA / name)))


def panel_a(ax):
    rows = {r["tissue"]: r for r in _load("run0_distribution.csv")}
    cols = ["<1", "1-5", "5-10", "10-15", ">=15"]
    x = np.arange(len(TIS)); bottom = np.zeros(len(TIS))
    for c, col, lab in zip(cols, STACK, SLAB):
        vals = np.array([float(rows[t][c]) for t in TIS])
        ax.bar(x, vals, bottom=bottom, color=col, width=0.72, label=lab, edgecolor="none")
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(TLAB)
    ax.set_ylabel("Fraction of sites"); ax.set_ylim(0, 1.0)
    ax.set_title("(a) Editing-level distribution", loc="left", fontsize=13)
    ax.legend(title="Editing level", loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=5, frameon=False, fontsize=9, title_fontsize=10, handlelength=1.2,
              columnspacing=1.1)


def _lines(ax, rows, colkeys, xvals, marker, title, ylabel, ylim, hline=None,
           legend_loc="upper right"):
    per = {v: {t: None for t in TIS} for v in VARS}
    for r in rows:
        if r["variant"] in VARS:
            per[r["variant"]][r["tissue"]] = [float(r[k]) for k in colkeys]
    for v in VARS:
        for t in TIS:                                   # faint per-tissue lines
            ax.plot(xvals, per[v][t], color=VCOL[v], lw=0.9, alpha=0.28, zorder=1)
        mean = np.nanmean([per[v][t] for t in TIS], axis=0)   # bold mean line
        ax.plot(xvals, mean, color=VCOL[v], lw=2.8, marker=marker, ms=8,
                zorder=3, label=v)
    if hline is not None:
        ax.axhline(hline, ls=":", lw=1.1, color="#666", zorder=0)
    ax.set_ylim(*ylim); ax.set_title(title, loc="left", fontsize=13)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc, frameon=False, fontsize=10)


def panel_b(ax):
    _lines(ax, _load("run1_score_by_bin.csv"),
           ["m_lt1", "m_1_5", "m_5_10", "m_10_15", "m_ge15"], np.arange(5), "o",
           "(b) Predicted score across the editing continuum", "Mean predicted score",
           (0.0, 1.0), hline=0.5, legend_loc="upper right")
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(BINS)
    ax.set_xlabel("Editing level (%)")


def panel_c(ax):
    _lines(ax, _load("run2_threshold_auroc.csv"), ["cut5", "cut10", "cut15"],
           [5, 10, 15], "s", "(c) Robustness to threshold relaxation", "AUROC",
           (0.80, 0.95), legend_loc="upper left")
    ax.set_xticks([5, 10, 15]); ax.set_xticklabels(["5", "10", "15"])
    ax.set_xlabel("Positive-label cutoff (%)")


def _finish(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    if name == "threshold_relaxation":
        MANUSCRIPT.mkdir(exist_ok=True)
        fig.savefig(
            MANUSCRIPT / "figS1_combined.png",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})

    # full 3-panel figure
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(18, 5.2))
    panel_a(a); panel_b(b); panel_c(c)
    fig.tight_layout()
    _finish(fig, "threshold_relaxation")

    # 2-panel figure (score continuum + threshold robustness)
    fig, (b, c) = plt.subplots(1, 2, figsize=(14, 5.2))
    # panel titles here are (a)/(b) in the 2-panel version
    panel_b(b); panel_c(c)
    b.set_title("(a) Predicted score across the editing-level continuum",
                loc="left", fontsize=13)
    c.set_title("(b) Robustness to threshold relaxation", loc="left", fontsize=13)
    fig.tight_layout()
    _finish(fig, "intermediate_site_scores")
