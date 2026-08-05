# SNP audit — does germline variation mimic or bias A-to-I editing?

The human substrates are reconstructed from the hg38 **reference** genome. This
analysis asks whether germline single-nucleotide polymorphism (SNP) — the fact
that individual donors differ from the reference — could masquerade as A-to-I
editing or bias the binary labels. Two complementary checks feed the two-panel
figure `figures/snp_audit.{png,pdf}`.

## Panel a — common-SNP burden per duplex

How many common germline SNPs fall inside each *Alu*-pair duplex, at the
population level.

- Mean **3.5** common SNPs per **~586-nt** duplex; **92.5%** of duplexes carry
  at least one; per-base rate ~0.59%.
- **`scripts/duplex_snp_burden.py`** — reads
  `data/raw/alu_pairs/Pair_Alu_withStrand.bed` (905 duplexes) and the UCSC hg38
  **Common SNPs 150** track (public; download to `data/raw/dbsnp/` — see the
  top-level README §3), and counts common SNPs per duplex. Writes
  `data/duplex_snp_burden_summary.json` and `data/duplex_snp_burden_hist.json`.

## Panel b — do donor genotypes actually mimic editing at labeled sites?

The population burden above says nothing about the *labeled adenosines*
themselves. Using the actual **GTEx v8 donor genotypes** (838 individuals,
whole-genome sequencing), we check how often a labeled site coincides with a
strand-aware editing-mimicking variant (A>G on the `+` strand, T>C on the `−`).

- Only **0.151%** of labeled sites overlap an editing-mimicking variant
  (**0.041%** at donor allele frequency ≥1%), and this is **balanced** across
  the edited (0.175%) and non-edited (0.145%) classes.
- The `donor_gtex_analysis/` sub-package holds this cascade (see its own
  `README.md`): `gtex_empirical_snp_summary.json` (the headline rates),
  `gtex_alu_site_inventory_summary.json` (the site denominator), and per-tissue
  / per-carrier breakdowns.

**Conclusion.** Donor genomic variation does not mimic A-to-I editing or bias
the binary labels — it affects a negligible, class-balanced fraction of sites.

## Other files

- **`scripts/make_snp_figure.py`** — assembles the two-panel figure from the
  burden JSONs (panel a) and the donor rates (panel b).
- **`scripts/real_dbsnp_audit.py`** — a base-resolution tally of common dbSNP
  variants at the labelled adenosine and at flanking control offsets; writes
  `data/real_dbsnp_audit_summary.json`. **This is not independent evidence and no
  number in the manuscript comes from it.** The labelled-site inventory it reads
  was itself built with the same dbSNP track applied at the queried position, so
  the count at offset 0 is 0 by construction, while flanking offsets sit at the
  genomic background rate (~0.55% per base, matching the 0.594% per-base density
  measured independently in panel a). The offset-0 figure therefore says nothing
  about biology. The manuscript's claim that donor variation does not mimic
  editing rests on the GTEx donor-genotype cascade in `donor_gtex_analysis/`
  (panel b), which uses actual donor genotypes rather than dbSNP and is not
  affected by this.
- `raw_data/` — the coordinate-resolved labeled-site inventory and the
  site×SNP overlap table used by the donor cascade.

## Reproduce

```bash
# Recreate the submitted figure from the shipped derived summaries and validate it:
bash run_all.sh

# Optional: recompute panel a upstream (needs the UCSC dbSNP track in data/raw/dbsnp/):
python scripts/duplex_snp_burden.py
```
The donor-genotype cascade (panel b) needs the controlled-access GTEx donor VCF
(dbGaP/AnVIL); its derived summaries are shipped in `donor_gtex_analysis/`.

`scripts/make_snp_figure.py` reads every plotted value from the shipped JSON
summaries; no figure value is hard-coded in the plotting script.
