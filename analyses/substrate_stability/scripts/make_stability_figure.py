#!/usr/bin/env python3
"""
Consolidated Supplementary figure for substrate structural stability:
 (a) intrinsic stability — distribution of the paired fraction across Alu substrates,
 (b) genomic-context maintenance — arm-arm paired fraction when each pair is re-folded
     with 200 bp of natural genomic flanking on each side,
 (c) held-out AUROC and (d) held-out F1 for the low- (<=Q1) vs high- (>=Q3) stability
     substrate cohorts, for both architectures across all five human settings.
"""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[3]
PERF = json.load(open(REPO / "analyses/substrate_stability/data/stability_performance.json"))
CTX = list(csv.DictReader(open(REPO / "analyses/substrate_stability/raw_data/context_folding_results.csv")))

G = PERF["_global"]
pf = np.array(G["paired_frac_values"]); Q1, Q3 = G["q1"], G["q3"]
dsrna = np.array([float(r["dsrna_frac"]) for r in CTX])

TISSUES = [("Artery", "Artery"), ("Brain", "Brain"), ("Liver", "Liver"),
           ("MuscleSkeletal", "Muscle"), ("Combined", "Combined")]
LOW, HIGH = "#E8A33D", "#3B6EA5"   # low-stability / high-stability cohort colours

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8})
fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
axa, axb, axc, axd = axes.flat


def letter(ax, ch):
    ax.text(-0.14, 1.10, ch, transform=ax.transAxes, fontweight="bold",
            fontsize=13, va="top", ha="left")


# (a) intrinsic stability distribution
axa.hist(pf, bins=40, color="#B9C4CE", edgecolor="white", linewidth=0.4)
axa.axvspan(pf.min(), Q1, color=LOW, alpha=0.18)
axa.axvspan(Q3, pf.max(), color=HIGH, alpha=0.18)
for x, c in [(Q1, LOW), (Q3, HIGH)]:
    axa.axvline(x, color=c, ls="--", lw=1.3)
axa.axvline(G["median"], color="#444", ls=":", lw=1.1)
axa.text(Q1, axa.get_ylim()[1]*0.96, f" Q1={Q1:.3f}", color=LOW, fontsize=8.5, ha="right", va="top")
axa.text(Q3, axa.get_ylim()[1]*0.96, f"Q3={Q3:.3f} ", color=HIGH, fontsize=8.5, ha="left", va="top")
axa.text(G["median"], axa.get_ylim()[1]*0.80, f"median\n{G['median']:.3f}", color="#444",
         fontsize=8, ha="center", va="top")
axa.set_xlabel("paired fraction (structural stability)")
axa.set_ylabel("number of substrates")
axa.set_title("Intrinsic substrate stability", fontsize=10, pad=10)
letter(axa, "a")

# (b) genomic-context maintenance
axb.hist(dsrna, bins=40, color="#B9C4CE", edgecolor="white", linewidth=0.4)
med = float(np.median(dsrna)); frac50 = 100*np.mean(dsrna >= 0.5)
axb.axvline(med, color="#444", ls=":", lw=1.1)
axb.text(med, axb.get_ylim()[1]*0.96, f" median {med:.2f}", color="#444", fontsize=8.5, ha="left", va="top")
axb.axvline(0.5, color=LOW, ls="--", lw=1.3)
axb.text(
    0.51,
    axb.get_ylim()[1] * 0.78,
    f"{frac50:.1f}% ≥ 0.5",
    color=LOW,
    fontsize=8.5,
    ha="left",
    va="top",
)
axb.set_xlabel("arm–arm paired fraction in genomic context (±200 bp)")
axb.set_ylabel("number of substrates")
axb.set_title("Hairpin maintenance in genomic context", fontsize=10, pad=10)
letter(axb, "b")


# (c,d) performance by stability cohort
def perf_panel(ax, metric, ylabel, title, ch, ylim):
    x = np.arange(len(TISSUES)); w = 0.19
    for k, (variant, off, hatch) in enumerate([("Baseline GAT", -1, None),
                                               ("Bio-aware GNN", 1, "////")]):
        for c, cohort, coff in [(LOW, "low_stability", -0.5), (HIGH, "high_stability", 0.5)]:
            vals = [PERF[t]["performance"][variant][cohort][metric] for t, _ in TISSUES]
            ax.bar(x + off*w + coff*w, vals, w, color=c, hatch=hatch,
                   edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in TISSUES], fontsize=9)
    ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10, pad=10)
    ax.grid(axis="y", ls=":", lw=0.4, alpha=0.5)
    letter(ax, ch)


perf_panel(axc, "auroc", "AUROC", "Discrimination by stability cohort", "c", (0.72, 0.97))
perf_panel(axd, "f1", "F1", "F1 by stability cohort", "d", (0.60, 0.93))

leg = [Patch(facecolor=LOW, label="low stability (≤Q1)"),
       Patch(facecolor=HIGH, label="high stability (≥Q3)"),
       Patch(facecolor="#CCCCCC", label="Baseline GAT"),
       Patch(facecolor="#CCCCCC", hatch="////", label="Bio-aware GNN")]

fig.tight_layout(rect=[0, 0.06, 1, 1])
# figure-level legend centred at the bottom, between panels c and d
fig.legend(handles=leg, fontsize=9, frameon=False, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, 0.005), handlelength=1.4, columnspacing=1.8)
out = REPO / "analyses/substrate_stability/figures/substrate_stability"
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
manuscript = REPO / "manuscript/fig_stability.png"
manuscript.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(out.with_suffix(".png"), manuscript)
print("wrote", out.with_suffix(".pdf"))
print("wrote", out.with_suffix(".png"))
print("wrote", manuscript)
