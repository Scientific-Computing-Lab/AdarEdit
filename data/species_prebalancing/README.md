# Cross-species pre-balancing tables

These gzip-compressed CSVs are the complete processed tables used immediately
before class balancing and region-disjoint train/validation/test splitting for
the three non-*Alu* species benchmarks.

They are **not the original source-study raw tables**. The tables have already
undergone the documented species data-construction workflow:

1. source editing-site processing;
2. editing-cluster construction;
3. dsRNA segment selection and RNAfold structure prediction;
4. candidate-adenosine expansion and editing-level merge;
5. duplicate removal, minimum dsRNA length of 200 nt, and minimum coverage of
   100 reads.

The next pipeline stage assigns the binary labels (`EditingLevel > 0.15` for
positive and `EditingLevel < 0.001` for operational negative), excludes the
boundary/intermediate sites, balances the classes, and assigns complete editing
clusters to disjoint splits.

## Files and integrity

`manifest.json` records the species, row count, compressed and uncompressed
byte counts, and SHA-256 digests. The Supplementary Fig. S5 validation checks
the compressed source hashes before accepting the figure.

The files can be inspected without extracting them:

```bash
gzip -cd data/species_prebalancing/Octopus_prebalancing.csv.gz | head
```

## Recompute Supplementary Fig. S5

From the repository root:

```bash
bash analyses/species_sensitivity/run_all.sh
```

The workflow reads these `.csv.gz` files directly, recomputes the full-dataset
distance distribution for panel a, recomputes the held-out test sensitivity
results for panels b--c, regenerates the figure, and validates all source
hashes, counts, metrics, and output files.
