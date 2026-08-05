#!/usr/bin/env python3
"""
Duplex-level SNP burden: how many common germline SNPs fall on each Alu-pair
double-stranded substrate (~600 nt) -- the quantity relevant to whether donor-vs-reference
variation could perturb the *predicted secondary structure* used as a model input.

Inputs:
  - alu_pairs.bed : 905 Alu pairs, columns chr,start1,end1,chr2,start2,end2,strand (hg38, BED 0-based).
  - UCSC hg38 Common Genomic SNPs (dbSNP150, MAF>=1%).

For each pair we count common SNVs within either arm [start1,end1) U [start2,end2),
giving a resource-based count of common SNVs in these Alu regions.
"""
import gzip, json, bisect, statistics as st
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BED = REPO / "data/raw/alu_pairs/Pair_Alu_withStrand.bed"
# EXTERNAL (public): UCSC hg38 Common SNPs 150 track -- download to data/raw/dbsnp/ (see README §3)
DBSNP = REPO / "data/raw/dbsnp/ucscHg38CommonGenomicSNPs150.bed.gz"
OUT = REPO / "analyses/snp_audit/data"

# ---- 1. load pairs, build per-chrom interval index -----------------------------
pairs = []
by_chrom = {}
with open(BED) as fh:
    for i, line in enumerate(fh):
        c1, s1, e1, c2, s2, e2, strand = line.split()
        s1, e1, s2, e2 = int(s1), int(e1), int(s2), int(e2)
        pairs.append({"chrom": c1, "len": (e1 - s1) + (e2 - s2), "snps": 0})
        by_chrom.setdefault(c1, []).append((s1, e1, i))
        by_chrom.setdefault(c2, []).append((s2, e2, i))
n_pairs = len(pairs)
idx = {}
for c, ivs in by_chrom.items():
    ivs.sort()
    idx[c] = (ivs, [v[0] for v in ivs])
print(f"pairs: {n_pairs} | duplex length mean {st.mean(p['len'] for p in pairs):.1f} nt "
      f"median {st.median(p['len'] for p in pairs)} nt", flush=True)

# ---- 2. stream dbSNP common SNVs, assign to containing arm ----------------------
scanned = hitting = 0
with gzip.open(DBSNP, "rt") as fh:
    for line in fh:
        scanned += 1
        f = line.split("\t")
        chrom = f[0]; s = int(f[1]); e = int(f[2])
        if e - s != 1:
            continue
        ent = idx.get(chrom)
        if ent is None:
            continue
        ivs, starts = ent
        j = bisect.bisect_right(starts, s) - 1
        while j >= 0 and starts[j] >= s - 700:
            st_, en_, pi = ivs[j]
            if st_ <= s < en_:
                pairs[pi]["snps"] += 1
                hitting += 1
                break
            j -= 1
print(f"dbSNP scanned {scanned}; common SNVs inside a duplex: {hitting}", flush=True)

# ---- 3. distribution ------------------------------------------------------------
counts = [p["snps"] for p in pairs]
with_ge1 = sum(c >= 1 for c in counts)
tot_len = sum(p["len"] for p in pairs)
res = {
    "resource": "UCSC hg38 Common Genomic SNPs dbSNP150 (MAF>=1%)",
    "n_pairs": n_pairs,
    "duplex_len_mean": st.mean(p["len"] for p in pairs),
    "duplex_len_median": st.median(p["len"] for p in pairs),
    "snps_per_duplex_mean": st.mean(counts),
    "snps_per_duplex_median": st.median(counts),
    "snps_per_duplex_max": max(counts),
    "pct_duplex_with_ge1_common_snp": 100.0 * with_ge1 / n_pairs,
    "pct_duplex_with_0_common_snp": 100.0 * (n_pairs - with_ge1) / n_pairs,
    "per_base_common_snp_density_pct": 100.0 * sum(counts) / tot_len,
}
print("\n==== common-SNP burden per Alu duplex (dbSNP150, MAF>=1%) ====")
print(f"mean SNPs / duplex     : {res['snps_per_duplex_mean']:.2f}")
print(f"median SNPs / duplex   : {res['snps_per_duplex_median']}")
print(f"max SNPs / duplex      : {res['snps_per_duplex_max']}")
print(f"duplexes with >=1 SNP  : {with_ge1}/{n_pairs} ({res['pct_duplex_with_ge1_common_snp']:.1f}%)")
print(f"duplexes with 0 SNP    : {n_pairs-with_ge1}/{n_pairs} ({res['pct_duplex_with_0_common_snp']:.1f}%)")
print(f"per-base common-SNP density: {res['per_base_common_snp_density_pct']:.3f}% / base")
(OUT / "duplex_snp_burden_summary.json").write_text(json.dumps(res, indent=2))
(OUT / "duplex_snp_burden_hist.json").write_text(json.dumps(dict(sorted(Counter(counts).items()))))
print("wrote", OUT / "duplex_snp_burden_summary.json")
