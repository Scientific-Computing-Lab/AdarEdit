# Cross-species sensitivity to the negative-labeling rule

In the three non-*Alu* species (octopus *O. bimaculoides*, acorn worm
*P. flava*, sea urchin *S. purpuratus*), candidate operational negatives were
collected by proximity to source-catalogued editing sites. This analysis tests
whether performance depends on the operational negatives that lie closest to a
benchmark-positive site (>15% editing).

## What this shows

Panel a uses the complete filtered and deduplicated **pre-balancing** tables in
`data/species_prebalancing/`. Operational negatives (`EditingLevel < 0.001`)
are measured against benchmark-positive sites (`EditingLevel > 0.15`) in the
same dsRNA region and arm; intermediate/boundary sites are not reference
positives, and every negative remains in the denominator.

For panels b--c, we take each species' within-species held-out **test** split and the
predictions of both architectures (Baseline GAT, Bio-aware GNN). Every negative
test adenosine is assigned a **proximity** = the minimum base distance to the
nearest >15% edited (positive) adenosine in the same reconstructed dsRNA region
and substrand. We then re-evaluate performance while **progressively excluding
the negatives closest to a positive** (within 5, 10, 30 bp).

Discrimination is retained across the whole sweep: AUROC remains broadly stable
or increases, with only small fluctuations, and F1 rises monotonically as the
nearest negatives are removed. Because removing negative examples at a fixed
threshold favors F1, the robustness interpretation is based primarily on the
threshold-free AUROC results.

| species | model | AUROC (all → >5 → >10 → >30 bp) | F1 (all → >5 → >10 → >30 bp) |
|---|---|---|---|
| *O. bimaculoides* | Baseline GAT | 0.865 → 0.865 → 0.868 → 0.868 | 0.809 → 0.813 → 0.819 → 0.835 |
| *O. bimaculoides* | Bio-aware GNN | 0.872 → 0.875 → 0.878 → 0.875 | 0.831 → 0.841 → 0.846 → 0.864 |
| *P. flava* | Baseline GAT | 0.823 → 0.836 → 0.845 → 0.853 | 0.791 → 0.811 → 0.831 → 0.859 |
| *P. flava* | Bio-aware GNN | 0.810 → 0.816 → 0.821 → 0.821 | 0.756 → 0.768 → 0.781 → 0.810 |
| *S. purpuratus* | Baseline GAT | 0.860 → 0.870 → 0.878 → 0.889 | 0.797 → 0.813 → 0.828 → 0.859 |
| *S. purpuratus* | Bio-aware GNN | 0.797 → 0.800 → 0.801 → 0.811 | 0.734 → 0.745 → 0.758 → 0.796 |

## Two protocol points

**Negatives whose region contains no benchmark-positive site are retained.**
A test negative whose region and arm contain no >15% site has no defined
proximity and is therefore not excluded by any minimum-distance cutoff. Such
negatives are 16–31% of the test negatives per species (octopus 232/749, acorn
worm 125/759, sea urchin 205/769), because the ≤20-nt anchor used during
candidate collection is *any source-catalogued editing site*, whereas the
benchmark-positive label additionally requires >15% editing.

**F1 uses the validation-selected threshold.** Each model's decision threshold
is read from its own checkpoint summary
(`checkpoints/{baseline,bioaware}_<species>/test_predictions.csv`: 0.600 / 0.325 for
octopus, 0.475 / 0.100 for acorn worm, 0.400 / 0.375 for sea urchin) rather than
swept over the evaluation set, matching the validation-only protocol used
throughout the paper. AUROC is threshold-free and does not use this decision
threshold.

## Files

- **`scripts/sensitivity.py`** — reads the canonical per-species predictions
  `checkpoints/{baseline,bioaware}_<species>/test_predictions.csv` and the
  species test splits, verifies row indices, labels and validation-selected
  thresholds, computes each negative's proximity, and writes the exclusion
  sweep (AUROC and F1 at each cutoff, per species × architecture) to
  `data/sensitivity.json`. SHA-256 hashes of every prediction and summary source
  are recorded in that JSON.
- **`scripts/compute_full_dataset_distances.py`** — reads the shipped compressed
  pre-balancing tables and recomputes the complete-dataset distance distribution
  and consecutive benchmark-positive spacing statistics in
  `raw_data/species_inter_site_distances.json`. In all three complete datasets,
  consecutive benchmark-positive sites in the same dsRNA region and arm have
  minimum spacing 1 bp and median spacing 4 bp.
- **`scripts/make_species_figure.py`** — reads `data/sensitivity.json` and
  `raw_data/species_inter_site_distances.json`, and writes
  `figures/species_sensitivity.{png,pdf}`: (a) full-dataset distance distribution
  of operational negatives to the nearest >15% benchmark-positive site,
  (b) AUROC and (c) F1 as the nearest negatives are progressively excluded.

Both scripts resolve their inputs and outputs relative to this folder, so the
analysis is self-contained.

## Reproduce

```bash
bash run_all.sh
```

This single command recomputes panel a from the shipped pre-balancing tables,
recomputes the test-set sensitivity table, creates the PNG/PDF, copies the
manuscript-ready PNG to `manuscript/species_sensitivity.png`, and validates all
counts, metrics, source hashes and figures.
