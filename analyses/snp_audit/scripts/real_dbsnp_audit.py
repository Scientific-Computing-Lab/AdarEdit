#!/usr/bin/env python3
"""
Empirical SNP audit at labeled GTEx Alu editing sites.

Inputs:
  - Coordinate-resolved GTEx Alu-site table (per site x tissue), giving for each
    of the 100,377 unique labeled sites: chromosome, 0-based PositionStart, transcript
    strand, and per-tissue edited/non-edited labels.
  - UCSC hg38 Common Genomic SNPs (dbSNP150, MAF>=1%), the same public resource the
    RNA Editing Index (AEI) uses to mask known polymorphisms.

Coordinate convention (verified base-by-base against the hg38 reference via Ensembl REST):
  1-based edited base = PositionStart + 1  (table is 0-based, BED-style)
  '+' strand site -> reference base A; editing-mimicking SNP = A>G (+ strand alleles {A,G})
  '-' strand site -> reference base T; editing-mimicking SNP = T>C (+ strand alleles {T,C})

For each common SNV we compute the base offset d = chromStart - PositionStart between the
SNV and every nearby labeled site (d = 0 is the edited base itself; d != 0 are flanking
control positions used to establish the local background common-SNP density in the same
Alu regions).
"""
import argparse, csv, gzip, json
from pathlib import Path

csv.field_size_limit(10**9)
REPO = Path(__file__).resolve().parents[3]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sites",
        required=True,
        help="Per site x tissue GTEx Alu-site label table "
             "(>1GB tissue-label CSV hosted externally — see repo README).",
    )
    return p.parse_args()


args = parse_args()
SITES = Path(args.sites)  # >1GB tissue-label table hosted externally — see repo README
# EXTERNAL (public): UCSC hg38 Common SNPs 150 track -- download to data/raw/dbsnp/ (see README §3)
DBSNP = REPO / "data/raw/dbsnp/ucscHg38CommonGenomicSNPs150.bed.gz"
OUTDIR = REPO / "analyses/snp_audit/data"
OUTDIR.mkdir(parents=True, exist_ok=True)
RC = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
FLANK = 5  # +/- bp window for background

# ---- 1. unique sites ------------------------------------------------------------
sites = {}   # (chr, PositionStart) -> {strand, pos_any}
with open(SITES, newline="") as fh:
    for row in csv.DictReader(fh):
        key = (row["PositionChr"], int(row["PositionStart"]))
        pos = row["is_positive_in_tissue"] == "True"
        s = sites.get(key)
        if s is None:
            sites[key] = {"strand": row["strand"], "pos_any": pos}
        elif pos:
            s["pos_any"] = True
n = len(sites); n_pos = sum(v["pos_any"] for v in sites.values()); n_neg = n - n_pos
print(f"unique sites {n} | edited {n_pos} | non-edited {n_neg}", flush=True)

for v in sites.values():
    v["exact_any"] = False; v["exact_mimic"] = False

# ---- 2. stream dbSNP common SNVs ------------------------------------------------
bg = {d: 0 for d in range(-FLANK, FLANK + 1)}      # matches at each offset (all sites)
scanned = 0
with gzip.open(DBSNP, "rt") as fh:
    for line in fh:
        scanned += 1
        f = line.rstrip("\n").split("\t")
        chrom, s, e, strand, observed = f[0], int(f[1]), int(f[2]), f[3], f[5]
        if e - s != 1:              # SNV only
            continue
        for d in bg:
            v = sites.get((chrom, s - d))
            if v is None:
                continue
            bg[d] += 1
            if d == 0:
                v["exact_any"] = True
                al = set()
                for a in observed.split("/"):
                    if len(a) == 1 and a in RC:
                        al.add(a if strand == "+" else RC[a])
                want = {"A", "G"} if v["strand"] == "+" else {"T", "C"}
                if al == want:
                    v["exact_mimic"] = True
print(f"dbSNP SNV scan done ({scanned} lines)", flush=True)

# ---- 3. tally -------------------------------------------------------------------
any_all = bg[0]
mim_all = sum(v["exact_mimic"] for v in sites.values())
any_pos = sum(v["exact_any"] and v["pos_any"] for v in sites.values())
mim_pos = sum(v["exact_mimic"] and v["pos_any"] for v in sites.values())
flank_vals = [bg[d] for d in bg if d != 0]
flank_mean = sum(flank_vals) / len(flank_vals)

def pc(a, b): return 100.0 * a / b if b else 0.0
print("\n==== REAL dbSNP150-common overlap at the EDITED base (strand-aware) ====")
print(f"edited base, any common SNV : {any_all}/{n}  ({pc(any_all,n):.4f}%)")
print(f"edited base, A>G/T>C mimic  : {mim_all}/{n}  ({pc(mim_all,n):.4f}%)")
print(f"  edited-label sites  : any {any_pos}/{n_pos} ({pc(any_pos,n_pos):.4f}%) | mimic {mim_pos}/{n_pos}")
print(f"  non-edited sites    : any {any_all-any_pos}/{n_neg} ({pc(any_all-any_pos,n_neg):.4f}%) | mimic {mim_all-mim_pos}/{n_neg}")
print(f"\nlocal background (flanking +/-1..{FLANK} bp): mean {flank_mean:.1f} SNPs/base  ({pc(flank_mean,n):.4f}% of sites/base)")
print("per-offset counts:", {d: bg[d] for d in sorted(bg)})

summary = {
    "resource": "UCSC hg38 Common Genomic SNPs dbSNP150 (MAF>=1%)",
    "resource_path": str(DBSNP), "genome": "hg38/GRCh38",
    "n_sites": n, "n_edited": n_pos, "n_non_edited": n_neg,
    "edited_base_any_common_snv": any_all,
    "edited_base_ag_mimic": mim_all,
    "edited_base_any_pct": pc(any_all, n),
    "flank_mean_per_base": flank_mean, "flank_pct_per_base": pc(flank_mean, n),
    "per_offset": {str(d): bg[d] for d in sorted(bg)},
}
(OUTDIR / "real_dbsnp_audit_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nwrote {OUTDIR/'real_dbsnp_audit_summary.json'}")
