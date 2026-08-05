# Data construction

How the benchmarks in `data/` were built, from raw editing tables to the
model-ready JSONL, plus a verifier for the pair-disjoint split.

Two complementary workflows are provided. To reproduce the reported analyses
and performance results, use the model-ready species datasets and checkpoints
included in this repository. To construct species datasets from the source
tables, use the data-construction pipeline, which validates focal coordinates,
nucleotide identity, labels and split assignments before generating model-ready
train, validation and test sets. Models trained from these generated datasets
can then be evaluated using the supplied analysis scripts.

```
data_construction/
├── human_alu/      Alu-pair substrates from hg38 + GTEx editing levels
├── species/        non-Alu substrates for the three distant species
├── split/          the global pair-disjoint train/valid/test protocol
├── add_pair_id.py  (re)generates the `pair_id` key in the metadata files
├── build_species_metadata.py  links species JSONL records to source provenance
├── verify_split.py            checks human pair/site overlap across splits
└── verify_species_split.py    checks species metadata and region disjointness
```

Paths are configurable via environment variables (`ADAREDIT_SITE_TABLES`,
`ADAREDIT_SPECIES_CSV`, `ADAREDIT_SPLIT_OUT`, `ADAREDIT_MODEL_DIR`); no absolute
paths are baked in. The large raw inputs (hg38 FASTA, per-tissue GTEx editing
tables, per-species editing tables from Zhang et al.) are external — see the
root `README.md` §3.

---

## 1. Human *Alu* substrates

### Inputs

`data/raw/alu_pairs/Pair_Alu_withStrand.bed` contains the 905 candidate inverted
*Alu* pairs. The five GTEx per-site tables are hosted externally because they
are approximately 700 MB:

```
Artery_Site_in_PairAlu_cov100.csv
Brain_Site_in_PairAlu_cov100.csv
Liver_Site_in_PairAlu_cov100.csv
MuscleSkeletal_Site_in_PairAlu_cov100.csv
Combined_Site_in_PairAlu_cov100.csv
```

Download them from the Google Drive folder linked in the root README and place
them, unchanged, in `data/raw/editing_levels/`. Each table contains the folded
substrate context (`structure`, `L`, `R`) and the continuous per-site
`EditingIndex`.

The Combined dataset was derived from the GTEx-wide table pooling data across
all 47 tissues. The precomputed Combined per-site editing levels are provided
in `Combined_Site_in_PairAlu_cov100.csv`.

`human_alu/Classification_Data_Creation.py` documents the upstream
sequence/structure construction: extract both arms from hg38, orient by strand,
join them with `NNNNNNNNNN`, fold with ViennaRNA and attach editing levels at
sites with more than 100 reads. Because the downloadable per-site tables
already contain these products, this expensive stage is not required to
reconstruct the reported benchmark.

### Label and balance before splitting

`human_alu/build_cross_splits.R` applies the binary benchmark definition:

- `yes`: `EditingIndex >= 15`;
- `no`: `EditingIndex < 1`;
- intermediate sites (`1 <= EditingIndex < 15`) are excluded;
- each context is balanced by downsampling the larger class before the
  whole-pair split.

Because random row selection can vary with R version and input order, exact
reconstruction does **not** resample the raw tables. Instead,
`human_alu/select_published_site_pool.py` derives the labels from the raw
`EditingIndex` values and selects the exact published record multiset using
`human_alu/published_site_selection.tsv.gz`. The manifest records the exact
pre-split balanced selection (24,000 Artery, 24,000 Brain, 19,690 Liver, 7,398
Muscle Skeletal and 24,000 Combined records). It stores only tissue, record
SHA256 and multiplicity; it contains no editing table or sequence.
`human_alu/make_selection_manifest.py` documents how this manifest was exported
from the authoritative within-tissue train/valid artifacts.

```bash
python human_alu/select_published_site_pool.py \
    --data-dir ../data/raw/editing_levels \
    --output-dir /tmp/adaredit_human_build/published_site_pools \
    --manifest human_alu/published_site_selection.tsv.gz \
    --yes-cutoff 15 --no-cutoff 1
```

### Apply the exact global pair split

`split/build_human_global_split.py` removes or rejects records whose full
sequence and dot-bracket lengths disagree (156 pre-split records are reported
and removed: 38/43/31/6/38 in Artery/Brain/Liver/Muscle
Skeletal/Combined). These are terminal adenosines for which CSV serialization
represented an empty left or right arm as `NA`, making the serialized
sequence two characters longer than its structure. It then assigns a
deterministic `pair_id` to each valid full `L + A + R` substrate and applies the
shipped `split/global_pair_split.json` to all five contexts. It writes the
model-ready JSONL and aligned metadata files.

```bash
python split/build_human_global_split.py \
    --balanced-dir /tmp/adaredit_human_build/published_site_pools \
    --output-dir /tmp/adaredit_human_build/reconstructed \
    --split-map split/global_pair_split.json \
    --invalid-policy drop \
    --expected-substrates 884 \
    --canonical-reference ../data/human
```

`--canonical-reference` performs an order-independent record comparison against
the shipped benchmark and fails if any site is missing, extra or assigned to a
different partition. It never modifies the reference files.

The complete three-stage command (label/balance, split/export, verify) is:

```bash
bash run_human_construction.sh \
    ../data/raw/editing_levels \
    /tmp/adaredit_human_reconstructed
```

The output directory must be empty or absent. All intermediate files are written
outside the shipped `data/human/` directory.

---

## 2. Cross-species substrates

The complete workflow is exposed through one command. Editing tables come from
Zhang et al., *Cell Reports* (2023). Create a CSV manifest with one row per
species (see `species_manifest.example.csv`):

```csv
species,editing_table,genome
Octopus,/data/octopus_editing_table.tsv,/refs/octopus.fa
Ptychodera,/data/ptychodera_editing_table.tsv,/refs/ptychodera.fa
Strongylocentrotus,/data/strongylocentrotus_editing_table.tsv,/refs/strongylocentrotus.fa
```

Run from the repository root, using a new or empty output directory:

```bash
bash data_construction/run_species_construction.sh \
    data_construction/species_manifest.csv \
    /tmp/adaredit_species_reconstructed \
    --num-processes 8 --merge-workers 8
```

The structure-segment step requires `bpRNA.pl` from
`https://github.com/hendrixlab/bpRNA`. Add it to `PATH`, set
`BPRNA_PL=/path/to/bpRNA.pl`, or append
`--bprna /path/to/bpRNA.pl` to the command. The Python/R environment is defined
in `environment.yml`; `dot2ct` and `draw` are not required for benchmark
construction because structure illustrations are not generated by default.

The final model-ready data are written to:

```text
/tmp/adaredit_species_reconstructed/benchmark/
├── Octopus/{train,valid,test}.jsonl
├── Ptychodera/{train,valid,test}.jsonl
└── Strongylocentrotus/{train,valid,test}.jsonl
```

Every JSONL has an aligned `*.metadata.csv` containing the genomic region,
source row, label and split. `species_split_summary.json` records the site and
region counts. `pipeline_provenance.json` records the resolved inputs,
parameters and commands. All intermediates remain under the requested output
directory; the repository data are never overwritten.

The workflow executes seven steps in order:

| # | script | what it does |
|---|---|---|
| 1 | `get_editing_levels.py` | parse the editing table → `A2IEditingSite.csv/.bed`; keeps A→G only, aggregates replicates, computes editing level and coverage |
| 2 | `cluster_editing_sites.py` | strand-aware merge of sites within 1 kb; keeps clusters with **>5** sites |
| 3 | `get_ds_with_majority_ES.py` | fold extended windows, pick the dsRNA segment holding the majority of editing sites, and collect nearby adenosines (**≤20 nt from a reported editing site**) as negative candidates |
| 4 | `merge_ds_results.py` | join the structures with per-site editing levels |
| 5 | `filter_ds_groups.R` | de-duplicate; keep `length ≥ 200` and `coverage ≥ 100` |
| 6 | `prepare_balanced_ml_sets.R` | apply strict labels, validate the focal adenosine, balance the complete classes and retain genomic provenance |
| 7 | `split/build_species_benchmark.py` | assign complete genomic regions to deterministic 64:16:20 train/validation/test partitions and write JSONL + metadata |

**Important — what "≤20 nt" anchors on.** Step 3 collects negatives within 20 nt
of *any reported editing site, at any editing level*. The **positive** label is
applied later (step 6) and additionally requires **>15 %** editing. A region whose
sites all fall below that cut therefore contributes negatives but no positives,
and most negatives sit further than 20 nt from the nearest >15 % site. This is why
`analyses/species_sensitivity/` measures distance to the nearest **>15 %** site
rather than to the labelling anchor.

```bash
Rscript species/prepare_balanced_ml_sets.R \
    --inputs "Species=/path/to/..._withoutDup.csv" \
    --out-dir balanced/ \
    --pos-threshold 0.15 --neg-threshold 0.001 \
    --equalize-across TRUE --seed 42 \
    --invalid-target-policy error

python split/build_species_benchmark.py \
    --balanced-root balanced/ \
    --out-dir benchmark/ \
    --seed 42 --fractions 0.64,0.16,0.20

python verify_species_split.py --data-root benchmark/
```

> `--pos-threshold` is **0.15** and the species preparation script applies it
> using a strict `>` comparison, matching the **>15 %** definition used for the
> cross-species benchmark. `--neg-threshold 0.001` is also strict (`<0.1 %`).
> Balancing is performed on the complete species pool before partitioning, as in
> the benchmark protocol.

### Focal-coordinate safety check

`Local_Position` is interpreted as one-based. The preparation stage rejects
missing, negative and out-of-range focal indices and checks that the
corresponding nucleotide in `small_ds_seq` is `A`; it never silently drops an
invalid index or replaces another nucleotide with an adenosine. The default
policy is `error`, so an upstream coordinate or strand problem stops the
workflow with source-row examples. `--invalid-target-policy drop` is available
only for sensitivity builds and should be paired with models trained from the
resulting dataset.

For direct reproduction of the reported species metrics and figures, use the
model-ready JSONL files and checkpoints supplied under `data/species/` and
`checkpoints/`. The source-data workflow provides a validated route from the
source tables to model-ready train, validation and test datasets, supporting
independent reconstruction and extension of the data-construction procedure.

The upstream structure script uses BED-style zero-based, half-open intervals
internally and writes one-based `Local_Position` values. It validates that each
selected arm length matches its mapped genomic interval and stores the RNAfold
structure of `small_ds_seq` itself (not the larger clustering window).

### Region-disjoint partition

The split unit is the source genomic cluster
`Chr:Strand:start_cluster:end_cluster`. All candidate adenosines from that
region remain in one partition. Region order is derived from SHA-256 of the
seed, species and region identifier, avoiding dependence on Python hash or RNG
implementation details. The validator asserts JSONL/metadata alignment,
sequence/structure length equality, both classes in every split and zero region
overlap.

---

## 3. The pair-disjoint split — `split/`

This directory implements the pair-disjoint splitting protocol
used for the reported benchmark.

| script | role |
|---|---|
| `build_human_global_split.py` | applies the shipped pair map to the five exact human site pools, writes JSONL + metadata, and can compare every reconstructed record with the reported benchmark |
| `build_global_split.py` | generates an alternative protocol-equivalent **whole-*Alu*-pair** 64:16:20 assignment with seed 42; use this only for a separately sampled benchmark, not to reproduce the manuscript map |
| `build_species_benchmark.py` | creates the model-ready region-disjoint species JSONL and aligned metadata |
| `global_pair_split.json` | the **authoritative** `pair_id`→split map that reproduces the reported results (566 train / 139 valid / 179 test), shipped next to the scripts |
| `export_global_split.py` | utility exporter for pre-existing JSONL inputs; it is not used by either canonical raw-data workflow |
| `check_u_handling.py` | guards the species alphabet: asserts `U` is encoded at the `T` position rather than falling through to `N` |

Because the assignment is made once per pair and shared across tissues, a
substrate is in the *same* partition in every tissue — so a site can never be in
one tissue's training set and another tissue's test set.

> **Reproducing the exact benchmark.** Exact site selection is fixed by
> `human_alu/published_site_selection.tsv.gz`, and the exact manuscript pair
> assignment is `global_pair_split.json`; do not regenerate or replace either
> file. The raw-data workflow above verifies the first against the downloaded
> tables and applies the second through `build_human_global_split.py`.
> `build_global_split.py` is provided for constructing an alternative,
> protocol-equivalent random assignment and will not reproduce the exact
> manuscript partition.

### What is shipped

The split **assignment itself** is shipped as data — the `split` column of every
`data/human/<tissue>/*.metadata.csv`. Every result in the
manuscript is computed on exactly that partition, so it can be inspected and
checked directly rather than only re-derived.

Class balancing is applied to each dataset as a whole before the pairs are
partitioned. Because whole pairs move together, the positive rate varies between
splits (45.6–56.5 %); `verify_split.py` reports it per tissue and split. AUROC and
AUPRC are prevalence-robust; F1 is not, so the per-split rate is worth knowing when
reading F1 columns.

---

## 4. Verify the shipped split

```bash
python verify_split.py
```

Expected output:

```
[1] substrate disjointness      : 884 substrates, 0 in more than one split
[2] cross-tissue site overlap   : worst A(train|valid)->B(test) cell = 0 (over 25 ordered tissue pairs)
[3] split proportions           : train=566 (64.0%)  valid=139 (15.7%)  test=179 (20.2%)
[4] substrate inventory         : 884 unique, 0 with length mismatch
[5] positive rate per split     : range 45.6%--56.5% positive
```

Checks 1–4 are assertions and the script exits non-zero if any fails. Check 5 is
reported, not asserted: because balancing was applied per dataset rather than per
split, and whole pairs move together, the positive rate varies between splits.
AUROC and AUPRC are prevalence-robust; F1 is not, so the per-split rate is worth
knowing when reading the F1 columns.
