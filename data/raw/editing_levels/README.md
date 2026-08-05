# GTEx per-site editing tables (download here)

This folder should contain one CSV per tissue:

```
Artery_Site_in_PairAlu_cov100.csv
Brain_Site_in_PairAlu_cov100.csv
Liver_Site_in_PairAlu_cov100.csv
MuscleSkeletal_Site_in_PairAlu_cov100.csv
Combined_Site_in_PairAlu_cov100.csv
```

**What they are.** Per-adenosine editing level and read coverage for every candidate *Alu* site in
each GTEx tissue — the raw source from which the binary edited (≥15%) / non-edited (<1%) labels in
`data/human/**/*.jsonl` were derived. Together ~700 MB.

**Who uses them.** The threshold-relaxation workflow
(`analyses/threshold_relaxation/scripts/build_full_cohorts.py` and
`run_analysis.py`) reads every row of these tables to derive the editing-level
distribution and the complete held-out 1--15% cohorts. They are also the
starting point for rebuilding the human `.jsonl` splits. The processed binary
splits are supplied in `data/human/`.

**How to obtain.** Too large to commit; hosted on Google Drive:
**https://drive.google.com/drive/folders/1KkGElOF-Peg0xzJWehONI5JfRKEAAdT_?usp=drive_link**

Download the five CSVs into this folder.

If the download service provides a large table as byte chunks named
`<CSV name>.part-*`, the threshold-relaxation builder streams all chunks
together in filename order before parsing them. They should not be parsed as
independent CSV files.

The Combined dataset was derived from the GTEx-wide table pooling data across
all 47 tissues. The precomputed Combined per-site editing levels are provided
in `Combined_Site_in_PairAlu_cov100.csv`.

After downloading, reconstruct and verify the human benchmark in a separate
output directory:

```bash
bash data_construction/run_human_construction.sh \
    data/raw/editing_levels \
    /tmp/adaredit_human_reconstructed
```

The command derives the labels from `EditingIndex`, verifies and selects the
exact published pre-split record multiset using
`data_construction/human_alu/published_site_selection.tsv.gz`, and then applies
the documented length-validity filter and the one global *Alu*-pair split
shared across all five contexts. It compares every reconstructed partition with
the released `data/human/` files and fails if any record is missing, extra or
assigned to the wrong split.
