# Independent-cohort label concordance

The model labels were derived from **GTEx** editing measurements. This analysis
compares them with independent measurements from **RNAAtlas**, which quantified
editing at the same *Alu* sites in different individuals using a distinct
sequencing and processing pipeline.

## Label concordance

For every *Alu* site measured in both cohorts we compare the GTEx editing index
to the independent RNAAtlas editing index.

- **Pearson r = 0.967**, **Spearman ρ = 0.932**, over **n = 16,028** shared
  sites.

The strong agreement between the two measurements supports the reproducibility
of the GTEx-derived binary labels across cohorts and processing pipelines.

- **`scripts/make_concordance_fig.py`** — reads
  `raw_data/Combined_GTEx_RNAatlas.csv` (per-site GTEx vs RNAAtlas editing
  index, coverage, and genomic coordinates) and writes
  `figures/rnaatlas_concordance.{png,pdf}` plus the correlation statistics.

## Files

- `raw_data/Combined_GTEx_RNAatlas.csv` — the paired GTEx/RNAAtlas per-site
  editing table (the concordance source).
- `figures/rnaatlas_concordance.{png,pdf}` — the concordance figure.

## Reproduce

From the repository root:

```bash
bash analyses/rnaatlas_external_cohort/run_all.sh
```

The runner writes the figure under `figures/` and prints the correlation
coefficients and number of co-measured sites.
