# Donor-level GTEx SNP audit — supporting tables

Compact per-site and summary tables quantifying how often a common germline variant — and
specifically an editing-mimicking A→G variant on the transcribed strand — falls on a labelled
adenosine in the actual GTEx donors whose RNA-seq defines the editing levels.

Every labelled site is cross-referenced against the GTEx v8 SHAPEIT2-phased donor genotypes
(controlled-access VCF; see the repository README). Because that genotype VCF is controlled-access
and cannot be redistributed, the derived summary tables below are provided in full; `required_inputs.json`
records the exact inputs and settings used to produce them.

**Tables:**
- `gtex_empirical_snp_summary.json` — top-level audit counts and rates (the numbers in the figure).
- `gtex_alu_site_inventory_summary.json` — inventory denominator (total labelled sites).
- `gtex_empirical_snp_tissue_summary.csv` — per-tissue breakdown.
- `gtex_empirical_snp_sample_summary.csv` — donor/sample aggregate counts.
- `gtex_empirical_snp_label_summary.tsv` — edited vs non-edited label-class breakdown.
- `gtex_empirical_snp_carrier_bins.tsv` — carrier-count bins.

**Key result:** of 100,377 labelled sites, only 1.51% carry any donor variant at the exact
coordinate and only 0.151% carry a strand-aware editing-mimicking variant (0.041% at donor
AF ≥ 1%, 0.030% at AF ≥ 5%); the strand-aware overlap is balanced across edited (0.175%) and
non-edited (0.145%) sites.

The full per-site tissue-label table (>1 GB) is not mirrored here.
