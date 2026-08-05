#!/usr/bin/env python3
"""Supplementary Fig. S4 from the two SNP analyses:
 a) dbSNP150-common burden across each ~586 nt Alu duplex  (general population; structure).
 b) GTEx v8 838-donor genotype overlap at the labeled sites (donor-specific; labels)."""
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
D = REPO / "analyses/snp_audit/data"
bd = json.loads((D / "duplex_snp_burden_summary.json").read_text())
hist = {int(k): v for k, v in json.loads((D / "duplex_snp_burden_hist.json").read_text()).items()}
donor = json.loads(
    (
        REPO
        / "analyses/snp_audit/donor_gtex_analysis/gtex_empirical_snp_summary.json"
    ).read_text()
)
overall = donor["overall_summary"]

CORAL = "#EE8A7F"; GREEN = "#5FA98C"; GREY = "#9AA3AB"; DARK = "#2F3B40"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                               gridspec_kw={"width_ratios": [1.25, 1.0]})

# ---- panel a: dbSNP common-SNP burden per duplex ----
xs = list(range(0, max(hist) + 1))
ys = [hist.get(x, 0) for x in xs]
axa.bar(xs, ys, color=GREEN, width=0.9, zorder=3)
axa.axvline(bd["snps_per_duplex_mean"], ls="--", lw=1.3, color=DARK, zorder=4)
axa.text(bd["snps_per_duplex_mean"] + 0.4, max(ys) * 0.9,
         f"mean {bd['snps_per_duplex_mean']:.1f}\nmedian {int(bd['snps_per_duplex_median'])}",
         fontsize=9, color=DARK, va="top")
axa.set_xlabel("common SNPs per Alu duplex")
axa.set_ylabel("number of duplexes")
axa.set_title("a", loc="left", fontweight="bold", fontsize=13)

# ---- panel b: GTEx 838-donor overlap at labeled sites ----
# Values are derived from the donor-level summary and expressed as percentages
# of the 100,377 labeled sites.
labels = ["any GTEx\nvariant", "strand-aware\nmimic", "GTEx\nAF≥1%", "GTEx\nAF≥5%"]
total_sites = int(overall["total_sites"])
vals = [
    100.0 * float(overall["any_coordinate_carrier_pct"]),
    100.0 * float(overall["transcript_equivalent_carrier_pct"]),
    100.0 * int(overall["transcript_equivalent_carrier_sites_af_ge_0_01"]) / total_sites,
    100.0 * int(overall["transcript_equivalent_carrier_sites_af_ge_0_05"]) / total_sites,
]
x = np.arange(len(vals))
axb.bar(x, vals, color=CORAL, width=0.66, zorder=3)
axb.set_yscale("log")
axb.set_ylim(0.02, 3)
for xi, v in zip(x, vals):
    axb.text(xi, v * 1.13, f"{v:.3f}%" if v < 1 else f"{v:.2f}%",
             ha="center", va="bottom", fontsize=8.4, color=DARK)
axb.set_xticks(x); axb.set_xticklabels(labels, fontsize=8.2)
axb.set_ylabel("labeled sites overlapping a\ndonor variant  [%, log]")
axb.set_title("b", loc="left", fontweight="bold", fontsize=13)

fig.tight_layout()
out = REPO / "analyses/snp_audit/figures/snp_audit"
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
manuscript = REPO / "manuscript/fig_snp.png"
manuscript.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(out.with_suffix(".png"), manuscript)
print("wrote", out.with_suffix(".pdf"))
print("wrote", out.with_suffix(".png"))
print("wrote", manuscript)
