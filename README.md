# AdarEdit

**Structure-aware Graph Learning Predicts RNA Editability Across Tissues and Species**

This repository, together with the linked external datasets, contains the model code,
processed data splits, trained best checkpoints, analysis inputs and outputs, and one
self-documenting folder per analysis. All reported numerical results are linked to supplied
artifacts and reproducible analysis workflows; external or controlled-access inputs needed
for upstream reconstruction are identified explicitly.

> **The central protocol.** All results use a **strict global pair-disjoint split**: every *Alu*
> inverted-repeat pair — and every adenosine site derived from it — is assigned to exactly one of
> train / validation / test (64:16:20), shared across all five tissues, with **zero pair overlap**
> between partitions. Class balancing is performed before partitioning; after the split is fixed,
> the test partition is not used for model, epoch or threshold selection.

## The two model architectures

Every model in this repository is one of two graph neural networks over the
predicted dsRNA secondary structure. They appear throughout as **Baseline GAT**
and **Bio-aware GNN**.

- **Baseline GAT** (`code/gnnadar_verb_compact.py`) — a compact graph attention
  network. Each nucleotide is a node with **8 features** (one-hot base A/G/C/U/N,
  a paired/unpaired flag, relative distance to the target adenosine, and a
  target-site flag); edges connect sequential (backbone) neighbours and
  base-paired nucleotides, and carry **no features** — the model receives only
  graph topology and must learn structural editing rules from data alone.
- **Bio-aware GNN** (`code/bioaware_gnn.py`) — the same graph enriched with
  explicit biochemistry: **22-dimensional node features** (adds 5′/3′ neighbour
  identity, stem/loop geometry, and base-pairing energy), **typed edges** that
  distinguish Watson–Crick (A-U, G-C) from wobble (G-U) and backbone edges (each
  with a learned embedding and scalar attributes), and a **parallel sequence-CNN
  branch** capturing local motifs. It tests whether hand-provided biological
  annotation improves prediction beyond the minimal baseline.

The **joint** multi-label model (one network predicting all five tissues at once)
comes in the same two flavours — **Joint Baseline** and **Joint Bio-aware** —
using these same two encoders (see `analyses/joint_model/`).

---

## 1. Requirements

```bash
conda env create -f environment.yml      # creates the 'adaredit' env (Python 3.10)
conda activate adaredit
# or manually:
pip install torch==2.7.0 torch-geometric==2.6.1 scikit-learn==1.6.1 \
            numpy==2.2.5 pandas==2.2.3 matplotlib \
            networkx shap seaborn pysam scipy xgboost
# ViennaRNA (RNAfold) is only needed to regenerate secondary structures from scratch:
#   conda install -c bioconda viennarna
```
A GPU is recommended for (re)training; all evaluation, the classical baselines and every figure
script run on CPU.

---

### Tested platform and installation time

The main training and evaluation pipeline was tested on Linux x86_64
(kernel 5.14, glibc 2.34) with Python 3.10, PyTorch 2.7.0,
PyTorch Geometric 2.6.1 and CUDA 12.8. The supplied checkpoints were
generated on an NVIDIA H200 GPU. Exact software and hardware provenance is
recorded in `environment.json` within each checkpoint directory. Analyses that
require a different software stack provide a dedicated environment file in
their analysis directory.

Creating the repository-level Conda environment typically takes approximately
15--30 minutes, depending on network speed and package availability. This
estimate does not include downloading the optional large raw GTEx tables.

## 2. Repository layout

```
ADAREDIT_repro/
├── README.md                       ← this file
├── environment.yml                 ← conda environment
├── code/                           ← model + trainer (self-contained)
│   ├── train_strict_long.py        ← exact per-context training pipeline used for the supplied checkpoints
│   ├── gnnadar_verb_compact.py     ← exact Baseline GAT implementation used for training
│   ├── bioaware_gnn.py             ← exact Bio-aware GNN implementation used for training
│   └── rna_attention_analysis.py   ← attention extraction utilities
├── data_construction/              ← how data/ was built, + split verifier (see §3.1)
│   ├── human_alu/                  ← Alu substrates from hg38 + GTEx editing levels
│   ├── species/                    ← non-Alu substrates for the three species (7 steps)
│   ├── split/                      ← the global pair-disjoint 64:16:20 protocol
│   └── verify_split.py             ← verifies pair/site disjointness across splits
├── data/
│   ├── human/{Artery,Brain,Liver,MuscleSkeletal,Combined}/{train,valid,test}.jsonl (+ .metadata.csv)
│   ├── species/{Octopus,Ptychodera,Strongylocentrotus}/{train,valid,test}.jsonl
│   └── raw/
│       ├── alu_pairs/Pair_Alu_withStrand.bed       ← 905 Alu-pair duplex definitions (hg38)
│       └── editing_levels/                          ← GTEx per-site editing tables — HOSTED ON GOOGLE DRIVE
│                                                       (see §3; too large to commit, ~700 MB)
├── checkpoints/                    ← BEST checkpoint + summary.json per model (see §7)
├── results/                        ← long-format results, manuscript table, matrices and predictions
├── analyses/                       ← one self-contained folder per analysis (see §6)
│   ├── threshold_relaxation/        ├── substrate_stability/
│   ├── gtex_tissue_selection/
│   ├── rnaatlas_external_cohort/    ├── component_ablation/
│   ├── snp_audit/                   ├── triplet_baseline/
│   ├── species_sensitivity/         ├── coding_targets/
│   ├── attention_interpretability/   ├── insilico_mutagenesis/
│   ├── joint_model/                 ├── joint_vs_pertissue/
│   └── minor_training_dynamics/
└── manuscript/                     ← the figures that appear in the paper (main + supplementary)
```
Each `analyses/<name>/` folder holds `scripts/` (code), `data/` and/or `raw_data/` (inputs and
derived tables), `figures/` (outputs) and usually its own `README.md`.

---

## 3. Data

**Processed data (in this repo).** Each `data/**/*.jsonl` line is one candidate adenosine as a
chat-style record. The user message encodes `L:<left seq>, A:<target base>, R:<right seq>,
Alu Vienna Structure:<RNAfold dot-bracket>`; the assistant message is `yes`/`no` (edited ≥15% /
non-edited <1%; intermediate 1–15% excluded). The full substrate is `L + A + R` and its length
equals the dot-bracket length.

Human `*.metadata.csv` files carry one row per site, aligned to the `.jsonl`, with
columns `pair_id, pair_prefix, yes_no, split`. **`pair_id` is the substrate key** — one
integer per distinct `L + A + R` duplex, shared across tissues, so a substrate has the same
id wherever it appears. Use it for any disjointness or grouping check. `pair_prefix` is the
first 24 nt of the duplex, useful for eyeballing but **not unique**: *Alu* elements share a
conserved 5' end, so 55 of the 884 substrates collide on it.

Species `*.metadata.csv` files are likewise aligned row-for-row to their JSONL files and
record the species, genomic `region_id`, source row, label, split, genomic coordinate,
strand, local position, editing level and source table. Use `region_id` to verify that
complete genomic regions remain disjoint across train, validation and test.

Pair_Alu_withStrand.bed contains 905 candidate inverted Alu pair loci. After
coverage/label selection and removal of 156 terminal-site records whose
serialized `L+A+R` length did not match the dot-bracket length,
884 graph-buildable full substrate structures remain. These are assigned stable
`pair_id` values 1..884 in all shipped human metadata files.

Test-split unique-site counts: Artery 4,792 · Brain 4,566 · Liver 4,150 ·
Muscle Skeletal 1,425 · Combined 4,864. All 884 retained substrates are
graph-buildable (dot-bracket length equals sequence length), and the global
pair-disjoint split assignment in the shipped metadata and
`data_construction/split/global_pair_split.json` contains exactly these 884
(566 train / 139 valid / 179 test).

**Raw data.**
- `data/raw/alu_pairs/Pair_Alu_withStrand.bed` — the 905 *Alu*-pair duplex loci (hg38, stranded).
- **GTEx per-site editing tables** `{Tissue}_Site_in_PairAlu_cov100.csv` (raw editing level and
  coverage for every candidate adenosine, the source from which the binary labels were derived).
  These are **~700 MB** and are hosted on Google Drive:
  **https://drive.google.com/drive/folders/1KkGElOF-Peg0xzJWehONI5JfRKEAAdT_?usp=drive_link**
  Download them into `data/raw/editing_levels/` to rebuild the `.jsonl` from scratch.
  The Combined dataset was derived from the GTEx-wide table pooling data across
  all 47 tissues; its precomputed per-site levels are provided in
  `Combined_Site_in_PairAlu_cov100.csv`.
- Analysis-specific raw inputs live inside each analysis folder's `raw_data/` (e.g. the RNAAtlas
  editing table, the donor-SNP overlap table) — see §6.

**Controlled-access / external data (NOT redistributed here).**
- GTEx v8 donor genotype VCF `GTEx_..._838Indiv_..._SHAPEIT2_phased.vcf.gz` (dbGaP/AnVIL) — used by
  the SNP donor cascade; (https://drive.google.com/file/d/1jVDpQ_AJ_X55TkCkZ4P5Z72uJ1FPuGPL/view?usp=drive_link)
- hg38 reference for coordinate/strand checks.
- **UCSC Common SNPs 150** track (public; ~143 MB) — the duplex SNP-burden script
  (`analyses/snp_audit/scripts/duplex_snp_burden.py`) reads it from
  `data/raw/dbsnp/ucscHg38CommonGenomicSNPs150.bed.gz`; download the track from the UCSC Genome
  Browser (hg38 `snp150Common`) into that folder to regenerate the panel-a numbers.

### 3.1 How the data was built — `data_construction/`

The upstream pipeline is included so the benchmark is auditable end to end
(`data_construction/README.md` has the full step-by-step):

| stage | folder | what it produces |
|---|---|---|
| Human *Alu* substrates | `data_construction/human_alu/` | labels the five downloaded per-site tables (`yes` ≥15 %, `no` <1 %, intermediate excluded), verifies the exact balanced pre-split site selection using the shipped manifest, and reports/removes 156 non-buildable terminal-site serializations |
| Cross-species substrates | `data_construction/species/` | seven steps from the Zhang et al. editing tables: parse → cluster (1 kb, >5 sites) → fold and pick the dsRNA segment → merge levels → filter (len ≥200, cov ≥100) → label and balance → region-disjoint split |
| Pair-disjoint split | `data_construction/split/` | applies the exact shipped `pair_id` map to whole *Alu* substrates at **64:16:20**, one assignment reused across all five contexts |

The cross-species stages, including the final region-disjoint 64:16:20 split,
can be run with one command:

```bash
bash data_construction/run_species_construction.sh \
    data_construction/species_manifest.csv \
    /tmp/adaredit_species_reconstructed \
    --num-processes 8 --merge-workers 8
```

Copy `data_construction/species_manifest.example.csv` and replace the editing
table and genome paths. The complete instructions, outputs and focal-adenosine
safety check are documented in `data_construction/README.md`. This workflow
also requires `bpRNA.pl`; provide it through `PATH`, `BPRNA_PL`, or the
`--bprna` option as described there.

To reconstruct the human benchmark after downloading the five tables:

```bash
bash data_construction/run_human_construction.sh \
    data/raw/editing_levels \
    /tmp/adaredit_human_reconstructed
```

The command writes only to the requested output and temporary directories. It
verifies the exact published site selection against the downloaded raw tables,
applies the immutable global pair assignment, compares every reconstructed
partition with `data/human/`, and then runs the split-overlap verifier.

**Verify the split yourself** — no external input needed:

```bash
python data_construction/verify_split.py
```

It asserts, directly from the shipped JSONL: 884 substrates with **0** appearing in
more than one split; **0** cross-tissue site overlap across all 25 ordered tissue pairs
(no site in one tissue's train/valid is in another tissue's test); and 64.0/15.7/20.2 %
partition proportions. It also reports the per-split positive rate (45.6–56.5 %),
which varies because balancing was applied to each dataset as a whole before splitting.

**Split provenance.** The exact split assignment is shipped as data — the `split` column
of every `data/human/<tissue>/*.metadata.csv` and
`data_construction/split/global_pair_split.json` — so the partition behind every
number in the paper is available directly.
`split/build_human_global_split.py` applies this exact map to reconstructed
balanced pools. `split/build_global_split.py` can create an alternative
protocol-equivalent random assignment, but it must not replace the manuscript
map. `verify_split.py` confirms that the shipped partition satisfies the
whole-pair, cross-context disjointness claims.

---

## 4. Train / evaluate a model

```bash
# One model (for example, Liver bio-aware). The output directory contains
# checkpoints/, history.csv, summary.json and held-out test predictions.
python code/train_strict_long.py \
  --variant bioaware \
  --context Liver \
  --data-root data/human \
  --cache-root cache/single \
  --out-root runs \
  --epochs 1000 \
  --batch-size 256 \
  --num-workers 0 \
  --num-threads 8 \
  --checkpoint-every 100 \
  --seed 42 \
  --resume
```

Use `--variant baseline|bioaware`; for species, replace `--data-root
data/human` with `--data-root data/species` and set `--context` to the species
directory name. A CUDA device is required by default; pass `--allow-cpu` only
for a CPU smoke test or an intentionally slow CPU training run.

This is the same validation-selected training implementation used to produce
the supplied per-context checkpoints. It records the input hashes, environment,
graph version and RNG state; writes checkpoints atomically; and refuses to
resume from an incompatible graph version. The default `--grad-clip 0.0`
matches the reported training recipe. The test split is evaluated only after
the requested final epoch, never for checkpoint or threshold selection.

To score with a provided checkpoint instead of retraining, load
`checkpoints/<model>/best.pth` (state under `model_state`). Each
`checkpoints/<model>/summary.json` holds the held-out `test_metrics` reported
in the paper.

---

### Training the Baseline GAT on a new dataset

To train the Baseline GAT on a new dataset, organize the input files as follows:

```text
<DATA_ROOT>/
└── <CONTEXT>/
    ├── train.jsonl
    ├── valid.jsonl
    └── test.jsonl
```

Here, `<DATA_ROOT>` is the directory containing the dataset and `<CONTEXT>` is
the name assigned to the new experimental context.

Each line of a JSONL file must describe one candidate adenosine using the
following format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Predict whether the central adenosine will be edited."
    },
    {
      "role": "user",
      "content": "L:<left sequence>, A:A, R:<right sequence>, Alu Vienna Structure:<dot-bracket structure>"
    },
    {
      "role": "assistant",
      "content": "yes"
    }
  ]
}
```

The assistant label must be `yes` for an edited site or `no` for a non-edited
site. The focal nucleotide must be `A`. The complete input sequence is formed
as `L + A + R`, and its length must equal the length of the corresponding
RNAfold dot-bracket structure.

The user is responsible for defining the biological criteria used to assign
the `yes` and `no` labels. If editing levels are available, intermediate or
uncertain sites may be excluded before creating the binary dataset.

Complete RNA substrates should be assigned to only one of the train,
validation or test partitions. All candidate sites derived from the same
sequence and predicted structure must remain in the same partition to prevent
substrate leakage.

Optional files named `train.metadata.csv`, `valid.metadata.csv` and
`test.metadata.csv` may be placed beside the JSONL files to retain site and
substrate provenance. These metadata files are not required for model training.
When supplied, each metadata file should contain one row for every corresponding
JSONL record and preserve the same row order.

For example, to train the Baseline GAT on a context named `MyDataset`, run from
the repository root:

```bash
python code/train_strict_long.py \
  --variant baseline \
  --context MyDataset \
  --data-root /path/to/new_data \
  --cache-root cache/new_data \
  --out-root runs \
  --epochs 1000 \
  --batch-size 256 \
  --num-workers 0 \
  --num-threads 8 \
  --checkpoint-every 100 \
  --seed 42
```

This command expects the following files:

```text
/path/to/new_data/MyDataset/train.jsonl
/path/to/new_data/MyDataset/valid.jsonl
/path/to/new_data/MyDataset/test.jsonl
```

During training, checkpoint and decision-threshold selection are performed
using the validation split only. After training is complete, the selected
checkpoint is evaluated once on the held-out test split. The test split is not
used for model, epoch or threshold selection.

The resulting files are written to:

```text
runs/baseline_MyDataset/
```

The output directory includes the selected model checkpoint, training history,
run configuration, recorded software environment, validation-selected
threshold, summary metrics and held-out test predictions.

A CUDA-enabled GPU is required by default for full training. The `--allow-cpu`
option may be added for a short smoke test or an intentionally slower CPU
training run:

```bash
python code/train_strict_long.py \
  --variant baseline \
  --context MyDataset \
  --data-root /path/to/new_data \
  --cache-root cache/new_data \
  --out-root runs \
  --epochs 1 \
  --batch-size 32 \
  --num-workers 0 \
  --num-threads 4 \
  --checkpoint-every 1 \
  --seed 42 \
  --allow-cpu
```

Full training time depends on the dataset size, number of epochs and available
GPU. The supplied checkpoints can be used to reproduce the reported analyses
without retraining.

## 5. Key verified results (held-out test)

Within-tissue (diagonal), Baseline GAT / Bio-aware GNN:

| Tissue | Base F1 | Base AUROC | Bio F1 | Bio AUROC |
|---|---:|---:|---:|---:|
| Artery | 0.867 | 0.924 | 0.858 | 0.918 |
| Brain | 0.869 | 0.930 | 0.863 | 0.927 |
| Liver | 0.854 | 0.909 | 0.854 | 0.910 |
| Muscle Skeletal | 0.814 | 0.869 | 0.816 | 0.878 |
| Combined | 0.867 | 0.933 | 0.836 | 0.903 |

Joint multi-label model (one model, all five tissue outputs): bio-aware
**macro F1 0.855 / AUROC 0.917**, baseline **0.851 / 0.911**. The no-Combined
control is documented in `analyses/joint_model/`. Triplet-SVM baseline (Liver):
**F1 0.785–0.793 / AUROC 0.784–0.787** across its three classifier heads.

---

## 6. Reproduce each analysis

Every folder is self-contained. Below: what it does, what to run, and what to expect. Each script
resolves its paths relative to its own location (`Path(__file__).resolve().parents[...]`), so the
analyses run in place with no path edits; controlled-access external inputs (donor VCF, hg38, UCSC
dbSNP, the large editing tables) are marked and documented in §3.

### threshold_relaxation — editing-level distribution & threshold robustness
- **Run:** from the repository root,
  `bash analyses/threshold_relaxation/run_all.sh`.
- **Inputs:** the five GTEx per-site editing tables under
  `data/raw/editing_levels/`, the authoritative human split, and the
  per-context checkpoints plus their canonical test predictions. The workflow
  derives the complete 1–15% held-out cohorts directly from these inputs.
- **Outputs:** `data/run0_distribution.csv`, `run1_score_by_bin.csv`, `run2_threshold_auroc.csv`;
  `data/analysis_metadata.json`,
  `figures/threshold_relaxation.*`, `figures/intermediate_site_scores.*`, and
  `manuscript/figS1_combined.png`.
- **Validation:** all 30,488 context-specific intermediate records must map to
  held-out substrates; all 60,976 model-site scores must align with their
  editing levels; local inference must reproduce canonical checkpoint
  predictions; and the ≥15% column must exactly equal each within-context test
  AUROC.

### gtex_tissue_selection — ADAR isoform expression in selected tissues
- **Run:** `bash analyses/gtex_tissue_selection/run_all.sh`.
- **Input:** a frozen 54-tissue snapshot of GTEx v8 median TPM for `ADAR`
  (ADAR1) and `ADARB1` (ADAR2), with API endpoint and Ensembl identifiers
  documented in the analysis README.
- **Outputs:** `figures/gtex_tissues.png/pdf` and
  `manuscript/gtex_tissues.png`.
- **Validation:** verifies the four plotted values against the frozen table,
  that Artery Tibial has the highest `ADARB1` across all GTEx v8 tissues, and
  that Muscle Skeletal has the lowest `ADAR`.

### rnaatlas_external_cohort — independent-cohort editing concordance
- **Run:** `bash analyses/rnaatlas_external_cohort/run_all.sh`.
- **Input (raw):** `raw_data/Combined_GTEx_RNAatlas.csv` — GTEx vs RNAAtlas editing index per site.
- **Outputs:** `figures/rnaatlas_concordance.png/pdf`.
- **Expect:** Pearson r = 0.967, Spearman ρ = 0.932 and n = 16,028
  co-measured sites. Of these, 1,678 fall in the intermediate RNAAtlas range;
  no site changes from GTEx non-edited to RNAAtlas edited, and one changes from
  GTEx edited to RNAAtlas non-edited.

### snp_audit — reference-vs-donor SNP audit
- **Run (figure from shipped summaries):**
  `bash analyses/snp_audit/run_all.sh`.
- **Optional upstream panel-a recomputation:** `python scripts/duplex_snp_burden.py` →
  `data/duplex_snp_burden_summary.json` (requires the public UCSC dbSNP track).
- **Panel b (donor cascade):** computed against the controlled-access GTEx v8 donor VCF; the derived
  summary tables are provided in `donor_gtex_analysis/` (see its README for the numbers).
- **Figure:** `python scripts/make_snp_figure.py`.
- **Input (raw):** `raw_data/gtex_empirical_snp_site_overlaps.csv` (100,377 site×donor overlaps).
- **Expect:** ~3.5 common SNPs per ~586-nt duplex, 92.5% carry ≥1; only **0.151%** of labelled
  sites carry a strand-aware editing-mimicking donor variant (0.041% at AF≥1%, 0.030% at AF≥5%).

### species_sensitivity — cross-species negative-distance sensitivity
- **Run:** `bash analyses/species_sensitivity/run_all.sh`.
- **Inputs:** `data/species/*`, `data/species_prebalancing/*.csv.gz`, and the
  within-species checkpoints and test predictions.
- **Outputs:** `data/sensitivity.json`; `figures/species_sensitivity.*`;
  `raw_data/species_inter_site_distances.json`; and
  `manuscript/species_sensitivity.png`.
- **Expect:** excluding operational negatives near benchmark-positive sites
  raises F1 (octopus baseline 0.809→0.835; bio-aware 0.831→0.864), while
  AUROC remains stable or increases modestly, supporting robustness to
  alternative minimum-distance criteria.

### substrate_stability — intrinsic & genomic-context structural stability
- **Run:** `bash analyses/substrate_stability/run_all.sh`.
- **Inputs:** `data/human/*` + checkpoints; `raw_data/context_folding_results.csv`,
  `raw_data/Pair_Alu_withStrand.bed`.
- **Outputs:** `data/stability_performance.json`;
  `figures/substrate_stability.png/pdf`; `manuscript/fig_stability.png`.
- **Expect:** paired-fraction Q1/median/Q3 = 0.776/0.812/0.844 over 884 substrates; genomic-context
  dsRNA fraction median 0.810 with 94.6% ≥0.5; least-stable-quartile AUROC ≈0.88–0.90
  in most contexts (Muscle Skeletal: 0.826–0.839).

### component_ablation — which bio-aware components carry signal (Liver)
- **Run from shipped outputs:** `bash analyses/component_ablation/run_all.sh`.
- **Retrain on GPU:** `bash analyses/component_ablation/scripts/run_ablations.sh`.
- **Inputs:** `data/human/Liver`; `checkpoints/bioaware_Liver/summary.json`;
  `checkpoints/ablations/Liver/*/summary.json`.
- **Outputs:** `ablation_summary.{csv,json}`;
  `figures/component_ablation.*`; `manuscript/figS_ablation.png`.
- **Expect (ΔAUROC vs full 0.9100):** neighbouring-base context −0.0540,
  pairing partner −0.0241, base-pair edges −0.0188, edge typing −0.0018,
  sequence-CNN −0.0054, and stem-loop geometry +0.0118. These are
  descriptive single-seed comparisons of separately trained models.

### triplet_baseline — Triplet-SVM / logistic-regression baseline (Xue 2005)
- **Run:** `bash analyses/triplet_baseline/scripts/run_all.sh Liver` (builds metadata and trains all three heads).
- **Input:** `data/human/Liver`.
- **Outputs:** `results/Liver_{logreg,linear_svm,rbf_svm}.json` (+ test predictions).
- **Expect:** F1 0.785–0.793 / AUROC 0.784–0.787 — below the graph models
  (AUROC 0.909–0.910).

### coding_targets — generalization to protein-coding editing targets
- **Run:** `bash analyses/coding_targets/run_all.sh`.
- **Inputs:** seven sequence/structure records and their Brain, Combined and Liver
  editing-level tables; six per-tissue checkpoints and two joint checkpoints.
- **Outputs:** 84 per-adenosine prediction files; `results/auroc_summary.{csv,json}`;
  joint-only and four-model AUROC panels; the FLNA structure panel; and
  `figures/coding_targets_full.{png,pdf}`.
- **Expect:** Joint Bio-aware AUROC is 0.86/0.83/0.85 for AJUBA,
  0.92/1.00/0.98 for BLCAP, 1.00/1.00/0.96 for FLNA, and
  0.90/0.91/0.93 for GRIA2 (Brain/Combined/Liver). The sparse NEIL1 and TTYH2
  cells remain descriptive. Interpret these per-gene AUROCs as a
  scope/generalization check rather than a definitive coding-target benchmark.

### attention_interpretability — attention analysis
- **Environment:** this analysis uses its own pinned software environment. From
  the repository root, create and activate it once with
  `conda env create -f analyses/attention_interpretability/environment.yml`
  followed by `conda activate adaredit-attention`. Use this environment rather
  than the repository-level environment for the attention workflow.
- **Run:** after activating `adaredit-attention`, execute
  `bash analyses/attention_interpretability/run_all.sh`.
- **Protocol:** extracts last-layer attention from the frozen Baseline Combined
  GAT across positions -50 to +50; fits and SHAP-ranks XGBoost probes on the
  validation split; evaluates them on the duplex-disjoint test split; and
  summarizes attention by editing label, nucleotide, and structural status
  across all test sites.
- **Inputs:** the checkpoint and Combined validation/test JSONL files included
  within the analysis directory.
- **Outputs:** `figures/attention_interpretability.png`, standalone panels B-G,
  complete metrics and predictions, node-level attention, SHAP data, and
  serialized XGBoost models.

### insilico_mutagenesis — in-silico mutagenesis (Fig 6)
- **Run:** `cd analyses/insilico_mutagenesis && bash run_all.sh`.
- **Inputs/model:** the self-contained Combined validation JSONL and
  Baseline Combined checkpoint stored in the analysis directory.
- **Outputs:** sample-level and summary CSV files under `data/`, individual
  A--D panels, and `figures/insilico_mutagenesis_full.{png,pdf}`.

### joint_model — joint multi-label training pipeline (both variants)
- **Validate supplied outputs:** `bash analyses/joint_model/run_all.sh`.
- **Retrain:** `bash analyses/joint_model/scripts/run_joint.sh`.
- **Models:** Baseline and Bio-aware, each with and without the Combined output.
- **Outputs:** four joint checkpoint directories under `checkpoints/` and the
  with/without-Combined control under
  `analyses/joint_model/results/combined_supervision_control.csv`.
- **Expect:** with-Combined macro F1 0.855 (bio-aware) / 0.851 (baseline);
  no-Combined macro F1 0.856 / 0.858.

### joint_vs_pertissue — joint vs five per-tissue models (Fig 2d)
- **Run:** `bash analyses/joint_vs_pertissue/run_all.sh`.
- **Inputs:** supplied per-tissue prediction NPZ files and joint summaries;
  `data/comparison_data.csv` is generated rather than manually entered.
- **Outputs:** `figures/joint_comparison_*` (slope + stacked).
- **Expect:** Bio-aware joint training improves the five-tissue macro F1 by
  0.009 and AUROC by 0.010; Baseline effects are mixed across tissues.

### minor_training_dynamics — training dynamics & validation-selected thresholds
- **Run:** `bash analyses/minor_training_dynamics/run_all.sh`.
- **Inputs:** the 16 per-context `history.csv`, `summary.json` and
  `run_config.json` files under `checkpoints/`.
- **Outputs:** `training_dynamics.png/pdf`,
  `training_dynamics_summary.csv/json`, and
  `manuscript/training_dynamics.png`.
- **Expect:** Baseline GAT threshold median 0.4625 (range 0.350--0.600);
  Bio-aware GNN median 0.2875 (range 0.100--0.400).

---

## 7. Checkpoints

```
checkpoints/
├── {baseline,bioaware}_{Artery,Brain,Liver,MuscleSkeletal,Combined}/best.pth   (10 human)
├── {baseline,bioaware}_{Octopus,Ptychodera,Strongylocentrotus}/best.pth        (6 species)
├── joint_{baseline,bioaware}/best.pth
└── joint_{baseline,bioaware}_noCombined/best.pth
```
Single-output summaries carry `test_metrics`; joint summaries carry
`test_by_tissue`, macro and pooled metrics. All use validation-selected
thresholds and report held-out test performance.

---

## 8. Reproducibility check (verified when assembled)

- All 20 best checkpoints are supplied: 16 single-output models and four joint
  multi-label models. Their summaries and saved predictions match the reported
  metrics.
- The Triplet-SVM baseline reproduces end-to-end **from the data in this repo** (F1 0.785 / AUROC 0.784).
- Every supplementary number was traced to the run-output file of its specific analysis
  (comprehensive metrics table, threshold relaxation, RNAAtlas concordance, SNP audit, cross-species
  sensitivity, substrate stability, component ablation).
- Precision/recall/AUPRC for off-diagonal cells live in the per-cell prediction files under the
  matrix package; the diagonal + thresholds are in each `checkpoints/*/summary.json`.
