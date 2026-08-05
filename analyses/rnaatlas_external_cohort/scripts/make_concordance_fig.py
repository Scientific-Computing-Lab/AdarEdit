#!/usr/bin/env python3
"""
Concordance figure (Nature Communications style): GTEx vs RNAAtlas
independently-measured Alu editing. Joint density scatter with marginal
editing-level distributions for each cohort.
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib import gridspec
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
OUT  = HERE.parent / "figures"; OUT.mkdir(exist_ok=True)
CSV  = HERE.parent / "raw_data" / "Combined_GTEx_RNAatlas.csv"

df = pd.read_csv(CSV, usecols=["GTEx_EditingIndex", "RNAatlas_EditingIndex"])
g = pd.to_numeric(df["GTEx_EditingIndex"], errors="coerce")
r = pd.to_numeric(df["RNAatlas_EditingIndex"], errors="coerce")
m = g.notna() & r.notna()
g, r = g[m].values, r[m].values
n = len(g)
pear = pearsonr(g, r)[0]; spear = spearmanr(g, r)[0]

GTEX_C = "#8c8c8c"     # neutral grey
RNA_C  = "#2b6cb0"     # blue
# clean white -> teal -> deep-blue density map
DENS = LinearSegmentedColormap.from_list(
    "dens", ["#f7fbff", "#c6dbef", "#6baed6", "#3182bd", "#08519c", "#08306b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 300,
})

fig = plt.figure(figsize=(88/25.4, 88/25.4))
gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5],
                       wspace=0.06, hspace=0.06)
ax  = fig.add_subplot(gs[1, 0])
axt = fig.add_subplot(gs[0, 0], sharex=ax)   # top marginal (GTEx)
axr = fig.add_subplot(gs[1, 1], sharey=ax)   # right marginal (RNAAtlas)

# ── main density panel ──────────────────────────────────────────────────────
hb = ax.hexbin(g, r, gridsize=48, norm=LogNorm(vmin=1), cmap=DENS,
               mincnt=1, linewidths=0.0)
ax.plot([0, 100], [0, 100], color="#c44e52", lw=0.8, ls=(0, (4, 3)), zorder=4)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xlabel("GTEx editing level (%)")
ax.set_ylabel("RNAAtlas editing level (%)")
ax.text(0.05, 0.95, f"Pearson $r$ = {pear:.2f}\nSpearman $\\rho$ = {spear:.2f}\n$n$ = {n:,} sites",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.8, linespacing=1.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ── marginals ───────────────────────────────────────────────────────────────
bins = np.linspace(0, 100, 61)
axt.hist(g, bins=bins, color=GTEX_C, alpha=0.9, lw=0)
axr.hist(r, bins=bins, color=RNA_C, alpha=0.9, lw=0, orientation="horizontal")
for a in (axt, axr):
    a.set_yscale("log") if a is axt else a.set_xscale("log")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
axt.spines["left"].set_visible(False); axt.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False)
axr.spines["bottom"].set_visible(False); axr.tick_params(bottom=False, labelbottom=False, labelleft=False, left=False)
axt.text(0.99, 0.72, "GTEx", transform=axt.transAxes, ha="right", va="center",
         color=GTEX_C, fontsize=7, fontweight="bold")
axr.text(0.60, 0.99, "RNAAtlas", transform=axr.transAxes, ha="left", va="top",
         color=RNA_C, fontsize=7, fontweight="bold", rotation=-90)

# ── slim colorbar inset inside main panel ───────────────────────────────────
cax = ax.inset_axes([0.62, 0.14, 0.30, 0.035])
cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
cb.outline.set_linewidth(0.4)
cb.set_label("sites per bin", fontsize=6, labelpad=2)
cb.ax.tick_params(labelsize=5.5, length=1.8, width=0.4)

stem = OUT / "rnaatlas_concordance"
fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
fig.savefig(str(stem) + ".png", dpi=400, bbox_inches="tight")
print(f"n={n:,}  pearson={pear:.4f}  spearman={spear:.4f}")
print("saved", stem.with_suffix(".png"))
