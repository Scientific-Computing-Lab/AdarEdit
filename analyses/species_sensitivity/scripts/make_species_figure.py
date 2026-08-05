#!/usr/bin/env python3
"""Supplementary cross-species negative-distance sensitivity figure.

(a) Full pre-balancing dataset distribution of operational negatives relative
    to the nearest benchmark-positive (>15%) site in the same dsRNA region and
    arm. Intermediate sites are not reference positives. (b) AUROC and (c) F1
    on the held-out test sets as negatives closest to a benchmark-positive site
    are progressively excluded. Baseline dashed/open, bio-aware solid/filled;
    models are not retrained.
"""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]   # this analysis folder, so a copied tree is self-contained
data = json.loads((HERE / "data/sensitivity.json").read_text())

SP_ORDER = ["Octopus", "Ptychodera", "S. purpuratus"]
SP_LABEL = {"Octopus": "O. bimaculoides", "Ptychodera": "P. flava", "S. purpuratus": "S. purpuratus"}
COL = {"Octopus": "#EE8A7F", "Ptychodera": "#5FA98C", "S. purpuratus": "#6C8EBF"}
BINS = ["≤5", "6–10", "11–20", "21–30", ">30 or no positive"]
BINCOL = ["#B02318", "#E0562B", "#F19A3E", "#F5C16C", "#D9DDE0"]

# Full-dataset negative -> nearest benchmark-positive (>15%) distribution.
dist_data = json.loads((HERE / "raw_data/species_inter_site_distances.json").read_text())
_name = {"Octopus bimaculoides": "Octopus", "Ptychodera flava": "Ptychodera",
         "Strongylocentrotus purpuratus": "S. purpuratus"}
DIST_HIST = {
    _name[sp["species"]]: {
        row["bin"]: row["fraction_all_negatives"]
        for row in sp["distance_bins"]
    }
    for sp in dist_data
}
XLAB = ["All\n(reported)", ">5 bp", ">10 bp", ">30 bp"]
X = list(range(4))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(13.2, 4.3),
                                    gridspec_kw={"width_ratios": [1.25, 1, 1]})

# ---- panel a: full distribution relative to a >=15% site ----
xs = np.arange(len(SP_ORDER))
bottom = np.zeros(len(SP_ORDER))
for b, color in zip(BINS, BINCOL):
    vals = np.asarray([DIST_HIST[s][b] * 100 for s in SP_ORDER])
    axa.bar(xs, vals, width=0.62, bottom=bottom, color=color,
            edgecolor="white", linewidth=0.5, zorder=3, label=b)
    bottom += vals
# Annotate the cumulative fraction within 20 bp.
for i, s in enumerate(SP_ORDER):
    within20 = sum(DIST_HIST[s][b] for b in BINS[:3]) * 100
    axa.text(i, 97, f"{within20:.1f}% ≤20 bp", ha="center", va="top",
             fontsize=7.4, color="#2F3B40")
axa.set_xticks(xs); axa.set_xticklabels([SP_LABEL[s] for s in SP_ORDER], fontsize=8, style="italic")
axa.set_ylabel("fraction of negative sites [%]")
axa.set_ylim(0, 109)
axa.set_title("a", loc="left", fontweight="bold", fontsize=13)
axa.legend(title="Distance to nearest >15% site (bp)", fontsize=7.0,
           title_fontsize=7.4, frameon=False, loc="upper center", ncol=2,
           handlelength=0.9, columnspacing=0.8, handletextpad=0.4,
           bbox_to_anchor=(0.5, 1.23))

# ---- panels b,c: sensitivity ----
for metric, ax, title, ylab in [("auroc", axb, "b", "AUROC"), ("f1", axc, "c", "F1")]:
    for sp in data:
        c = COL[sp["short"]]
        for model, ls, fill in [("Baseline GAT", "--", "none"), ("Bio-aware GNN", "-", c)]:
            ys = [r[metric] for r in sp["results"][model]]
            ax.plot(X, ys, ls=ls, color=c, marker="o", markersize=5,
                    markerfacecolor=fill, markeredgecolor=c, lw=1.6, zorder=3)
    ax.axvline(0, color="#999", lw=1.0, ls=":", zorder=1)
    ax.set_xticks(X); ax.set_xticklabels(XLAB, fontsize=8.5)
    ax.set_xlim(-0.3, 3.3)
    ax.set_xlabel("negatives retained (min. dist. to a >15% site)")
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=13)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5)

sp_handles = [Line2D([0], [0], color=COL[s], lw=2.6, label=SP_LABEL[s]) for s in SP_ORDER]
mdl_handles = [Line2D([0], [0], color="#555", ls="--", marker="o", markerfacecolor="none",
                      markeredgecolor="#555", label="Baseline GAT"),
               Line2D([0], [0], color="#555", ls="-", marker="o", markerfacecolor="#555",
                      label="Bio-aware GNN")]
fig.legend(handles=sp_handles, loc="upper center", bbox_to_anchor=(0.55, 0.03),
           ncol=3, fontsize=8.5, frameon=False, title="Species", title_fontsize=8.5)
fig.legend(handles=mdl_handles, loc="upper center", bbox_to_anchor=(0.90, 0.03),
           ncol=1, fontsize=8.5, frameon=False, title="Model", title_fontsize=8.5)

fig.tight_layout(rect=[0, 0.11, 1, 1])
FIGDIR = HERE / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"species_sensitivity.{ext}", bbox_inches="tight")
manuscript = REPO / "manuscript/species_sensitivity.png"
manuscript.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(FIGDIR / "species_sensitivity.png", manuscript)
print("wrote", FIGDIR / "species_sensitivity.pdf")
print("wrote", FIGDIR / "species_sensitivity.png")
print("wrote", manuscript)
