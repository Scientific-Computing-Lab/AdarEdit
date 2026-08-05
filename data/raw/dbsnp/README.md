# dbSNP common-variant track (download here)

This folder should contain one file:

```
ucscHg38CommonGenomicSNPs150.bed.gz
```

**What it is.** The UCSC hg38 **Common SNPs (150)** track (`snp150Common`) in BED format —
germline variants with minor-allele frequency ≥ 1% in at least one population. ~143 MB, public.

**Who uses it.** `analyses/snp_audit/scripts/duplex_snp_burden.py` reads this file to count common
SNPs falling within each *Alu*-pair duplex (Supplementary SNP figure, panel a: mean 3.5 common SNPs
per ~586-nt duplex, 92.5% carry ≥1). The derived result is already provided in
`analyses/snp_audit/data/duplex_snp_burden_summary.json`; this file is only needed to **regenerate**
those numbers from scratch.

**How to obtain.** It is a public resource; it is not redistributed here. Options:
- UCSC Genome Browser → Table Browser: assembly **hg38**, group *Variation*, track *Common SNPs(150)*,
  table `snp150Common`; export as BED and gzip to the filename above; or
- take `snp150Common.txt.gz` from `hgdownload.soe.ucsc.edu/goldenPath/hg38/database/` and convert the
  chrom/chromStart/chromEnd columns to a BED; or
- use the copy bundled with the RNAEditingIndex (AEI) resource pack, which is this exact file.

Place the resulting `ucscHg38CommonGenomicSNPs150.bed.gz` in this folder and re-run the script.
