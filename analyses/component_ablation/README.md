# Component ablation of the bio-aware model

This directory provides one command to reproduce Supplementary Fig. S7 from
the Liver outputs included in the repository. It also provides an
optional GPU script for retraining the full reference model and all six
ablations from scratch.

All paths below are relative to the repository root. The analysis deliberately
uses the single canonical copies of the Liver data and model outputs stored at
repository level; data and checkpoints are not duplicated inside this
directory.

## What the analysis tests

The full bio-aware Liver model is compared with six models in which
one component is changed:

| tag | model change |
|---|---|
| `no_neighbors` | remove the explicit 5′/3′ neighbouring-base node features |
| `shuffle_pair_edges` | retain paired-node degree but replace true pairing partners with deterministic random partners |
| `no_pair_edges` | remove structural base-pair edges while retaining backbone edges |
| `untyped_edges` | collapse the categorical backbone/canonical/wobble/other-pair edge labels to one label |
| `no_seq_branch` | remove the parallel sequence-CNN branch |
| `no_geometry` | remove stem-length, loop-length and junction-distance features |

The ablations are not post-hoc masks applied to one checkpoint. The full model
and every ablated model are trained independently from initialization.

All seven models use:

- the same pair-disjoint Liver train/validation/test split;
- seed 42;
- 1,000 requested training epochs;
- checkpoint and decision-threshold selection using validation F1 only; and
- one final evaluation on the held-out test set after model selection.

This is a single-seed, descriptive ablation analysis. It does not estimate
between-seed uncertainty or statistical significance.

## Inputs

The analysis uses the following files from the repository:

```text
repository-root/
├── environment.yml
├── data/human/Liver/
│   ├── train.jsonl             # 12,485 sites
│   ├── valid.jsonl             #  3,024 sites
│   └── test.jsonl              #  4,150 sites
├── checkpoints/
│   ├── bioaware_Liver/
│   │   ├── best.pth
│   │   ├── summary.json
│   │   └── test_predictions.*
│   └── ablations/Liver/
│       ├── no_neighbors/
│       ├── shuffle_pair_edges/
│       ├── no_pair_edges/
│       ├── untyped_edges/
│       ├── no_seq_branch/
│       └── no_geometry/
└── analyses/component_ablation/
    ├── pipeline/               # compatibility launchers for the canonical repository code
    ├── scripts/                # preflight, training, summary and plotting
    ├── run_all.sh               # reproduce the reported table and figure
    └── README.md
```

`scripts/check_inputs.py` stops immediately with an explicit missing-file
message if the required Liver data or checkpoints are unavailable.

## Software environment

Create the supplied Conda environment once, from the repository root:

```bash
conda env create -f environment.yml
conda activate adaredit
```

The reported models were produced with Python 3.10, PyTorch 2.7.0,
PyTorch Geometric 2.6.1, scikit-learn 1.6.1, NumPy 2.2.5 and pandas 2.2.3.
For GPU retraining, install the PyTorch build appropriate for the local CUDA
driver if the default environment does not expose CUDA.

## Reproduce the reported table and figure

Use this route to verify the shipped results and regenerate
Supplementary Fig. S7. It does not retrain models and does not require a GPU.

From the repository root:

```bash
conda activate adaredit
cd analyses/component_ablation
bash run_all.sh
```

The command performs the following steps:

1. verifies that the canonical data, full model and six ablation checkpoints
   are present;
2. verifies the graph version, data hashes, split sizes, seed, epoch budget and
   validation-only selection protocol recorded in the seven summaries;
3. regenerates `ablation_summary.json` and `ablation_summary.csv`;
4. regenerates `figures/component_ablation.png` and `.pdf`;
5. copies the manuscript-ready PNG to
   `../../manuscript/figS_ablation.png`; and
6. verifies the reported metrics, source hashes and equality of the two PNG
   copies.

Successful completion ends with:

```text
PASS: all shipped component-ablation inputs are present
PASS: all seven runs record the graph invariants
PASS: Liver component-ablation validation
[component ablation] PASS
```

Generated or refreshed files:

```text
analyses/component_ablation/ablation_summary.csv
analyses/component_ablation/ablation_summary.json
analyses/component_ablation/figures/component_ablation.png
analyses/component_ablation/figures/component_ablation.pdf
manuscript/figS_ablation.png
```

## Optional: retrain the full model and all six ablations

Use this route for a complete GPU reproduction from the shipped Liver split.
It trains seven independent models: the full bio-aware reference plus all six
ablations.

From the repository root:

```bash
conda activate adaredit
cd analyses/component_ablation
bash scripts/run_ablations.sh
```

By default the script uses:

```text
epochs          1000
batch size      256
data workers    0
CPU threads     8
seed            42
```

The data are read from:

```text
data/human/Liver/{train,valid,test}.jsonl
```

New graph caches and runs are written only under this analysis:

```text
analyses/component_ablation/cache/
analyses/component_ablation/recomputed_runs/bioaware_Liver/
analyses/component_ablation/recomputed_runs/ablations/Liver/<tag>/
```

The trainer writes, for every model:

```text
best checkpoint
last/resume checkpoint
training history
run configuration
environment record
test predictions
ROC/PR curve arrays
summary metrics
```

After all seven trainings finish, the script sets the newly trained full model
and ablations as the analysis sources, regenerates the summary and figure, and
runs validation. It does not silently fall back to the shipped full model.

Training is resumable. Re-running the same command continues from each
model's `last.pth` checkpoint. Optional settings may be passed as environment
variables:

```bash
EPOCHS=1000 \
BATCH_SIZE=256 \
NUM_WORKERS=0 \
NUM_THREADS=8 \
PYTHON_BIN=python3 \
bash scripts/run_ablations.sh
```

For a publication reproduction, retain the defaults. Reducing `EPOCHS` is
useful only for a smoke test and will not reproduce the reported results.

## Reported Liver results

The full bio-aware reference reaches F1 = **0.8543**, AUROC = **0.9100** and
AUPRC = **0.9269**.

| model change | F1 | AUROC | ΔF1 | ΔAUROC |
|---|---:|---:|---:|---:|
| no neighbouring-base context | 0.7909 | 0.8560 | −0.0634 | −0.0540 |
| shuffled pairing partners | 0.8256 | 0.8859 | −0.0287 | −0.0241 |
| no base-pair edges | 0.8357 | 0.8912 | −0.0187 | −0.0188 |
| no sequence-CNN branch | 0.8483 | 0.9046 | −0.0060 | −0.0054 |
| untyped edges | 0.8425 | 0.9082 | −0.0118 | −0.0018 |
| no stem-loop geometry | 0.8598 | 0.9218 | +0.0055 | +0.0118 |

Deltas are calculated as ablated minus full-model test performance. Negative
values therefore indicate lower performance after changing the component.

The largest reductions occur after removing neighbouring-base context,
randomizing the pairing partners or removing base-pair edges. Categorical edge
typing and the sequence-CNN branch have smaller observed effects. Removing the
geometry features slightly improves this single Liver run, so this experiment
does not provide evidence that the hand-crafted geometry features improve
Liver prediction.

## Code-to-output map

| file | role |
|---|---|
| `pipeline/bioaware_gnn.py` | compatibility import for `../../code/bioaware_gnn.py` |
| `pipeline/gnnadar_verb_compact.py` | compatibility import for `../../code/gnnadar_verb_compact.py` |
| `pipeline/train_strict_long.py` | compatibility launcher for `../../code/train_strict_long.py` |
| `scripts/check_inputs.py` | preflight check for canonical data and checkpoints |
| `scripts/run_ablations.sh` | full seven-model GPU retraining workflow |
| `scripts/summarize_ablations.py` | provenance checks and summary-table generation |
| `scripts/make_ablation_fig.py` | Supplementary Fig. S7 generation |
| `scripts/validate_outputs.py` | final metric, hash and figure validation |
| `run_all.sh` | one-command reproduction from shipped outputs |

No script in this analysis contains a machine- or user-specific absolute path.
