# Supplementary Figure S1: editing-level continuum and threshold relaxation

This directory contains the complete workflow used to generate Supplementary
Figure S1. It tests whether models trained on the binary editing classes
(`<1%` and `>=15%`) assign progressively higher scores across intermediate
editing levels and remain discriminative when the positive-label cutoff is
relaxed.

The workflow performs inference with the supplied checkpoints. It does not
train models, reselect checkpoints, or optimize thresholds on the test set.

## Figure panels

Supplementary Figure S1 contains:

1. the editing-level distribution before binary-label filtering;
2. mean scores across `<1%`, `1--5%`, `5--10%`, `10--15%`, and `>=15%`; and
3. held-out AUROC with positive-label cutoffs of 5%, 10%, and 15%.

The analysis includes Artery Tibial, Brain Cerebellum, Liver, Muscle Skeletal,
and Combined, for both the baseline and bio-aware models.

## Requirements

Create the repository environment from the repository root:

```bash
conda env create -f environment.yml
conda activate adaredit
```

The checkpoint-inference environment is specified in `environment.yml`,
including Python 3.10, PyTorch 2.7.0, torch-geometric 2.6.1, NumPy 2.2.5,
pandas 2.2.3, and scikit-learn 1.6.1. A CUDA-capable GPU is recommended; CPU
inference is supported but slower.

## Source editing-level tables

Place the five GTEx per-site editing tables in `data/raw/editing_levels/`:

```text
Artery_Site_in_PairAlu_cov100.csv
Brain_Site_in_PairAlu_cov100.csv
Liver_Site_in_PairAlu_cov100.csv
MuscleSkeletal_Site_in_PairAlu_cov100.csv
Combined_Site_in_PairAlu_cov100.csv
```

The download location is documented in `data/raw/editing_levels/README.md`.
For storage systems that split a large CSV into byte chunks, files named
`<CSV name>.part-*` are also supported. All parts are streamed together in
lexicographic order before CSV parsing, so records that cross chunk boundaries
are retained.

## Run

From the repository root:

```bash
bash analyses/threshold_relaxation/run_all.sh
```

All paths are resolved relative to the repository. Optional settings are:

```bash
PYTHON_BIN=python \
ADAREDIT_THRESHOLD_DEVICE=cuda \
ADAREDIT_THRESHOLD_WORK_DIR=/path/to/temporary/work \
bash analyses/threshold_relaxation/run_all.sh
```

`ADAREDIT_THRESHOLD_DEVICE` may be `auto`, `cpu`, or `cuda`. Temporary graph
caches are written under `TMPDIR` by default. To read the source editing tables
from another location without changing the code, set
`ADAREDIT_EDITING_LEVEL_DIR` to that directory.

By default, all intermediate-site predictions are regenerated. Set
`ADAREDIT_THRESHOLD_RESUME=1` only to reuse complete prediction CSVs already
present in `data/intermediate_predictions/`; all provenance and downstream
validation are still refreshed.

## Analysis procedure

1. `build_full_cohorts.py` reads every row of each source table and calculates
   the model-independent distribution shown in panel a.
2. It reconstructs the global substrate-to-split assignment from
   `data/human/{Context}/{train,valid,test}.jsonl` and retains intermediate
   sites only when their complete Alu-pair substrate belongs to the test split.
3. Sites with editing levels in `[1%,15%)` are deduplicated by left sequence,
   right sequence, and structure. The resulting held-out intermediate cohorts
   contain 30,488 context-specific records:

   | Context | Records |
   |---|---:|
   | Artery | 6,638 |
   | Brain | 6,122 |
   | Liver | 4,893 |
   | Muscle Skeletal | 4,549 |
   | Combined | 8,286 |

4. `run_analysis.py` first reconstructs 64 canonical test predictions for each
   context and architecture and compares them with the supplied checkpoint
   predictions. Labels must match exactly; a small absolute probability
   tolerance permits numerical variation between supported CPU/GPU backends.
5. The supplied baseline and bio-aware checkpoints score every intermediate
   site, without retraining.
6. Intermediate scores are combined with the canonical `<1%` negatives and
   `>=15%` positives from each checkpoint's held-out predictions.
7. Mean scores are calculated for the five editing-level bins, and AUROC is
   calculated at positive cutoffs of 5%, 10%, and 15%.

For a cutoff `c`, negatives remain the canonical `<1%` sites, positives are
sites with editing level `>=c`, and sites in `[1%,c)` are excluded. The 15%
condition is therefore exactly the binary held-out test AUROC supplied with the
checkpoint.

## Outputs

- `data/cohorts/`: complete held-out intermediate cohorts and editing levels;
- `data/intermediate_predictions/`: per-context and per-model predictions with
  provenance metadata;
- `data/intermediate_scores.csv`: all 60,976 model-site scores;
- `data/run0_distribution.csv`: editing-level distributions;
- `data/run1_score_by_bin.csv`: mean scores by editing-level bin;
- `data/run2_threshold_auroc.csv`: AUROC at the three positive cutoffs;
- `data/cohort_provenance.json` and `data/analysis_metadata.json`: input,
  environment, fidelity, and output provenance;
- `figures/threshold_relaxation.{png,pdf}`: complete three-panel figure;
- `figures/intermediate_site_scores.{png,pdf}`: panels b and c; and
- `manuscript/figS1_combined.png` and `manuscript/threshold_relaxation.png`:
  manuscript-ready copies.

## Validation

`validate_outputs.py` runs automatically. It verifies cohort counts and ranges,
input and prediction hashes, prediction-to-level row alignment, all ten
context/model groups, monotonic mean score bins, non-decreasing AUROC across
the three cutoffs, exact equality of each 15% AUROC to its checkpoint summary,
canonical checkpoint-inference fidelity, and all expected figures.

Successful completion ends with:

```text
PASS: threshold-relaxation analysis finished.
```
