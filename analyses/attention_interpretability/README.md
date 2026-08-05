# Attention interpretability of the Baseline Combined GAT

This directory reproduces the attention-analysis panels for the frozen
Baseline Combined graph-attention model. It contains the model checkpoint,
validation and test inputs, analysis code, derived data, serialized XGBoost
probes, and final figures required to run the analysis without external data.

## Analysis design

The analysis uses attention coefficients from the last GAT layer (layer 2).
For each candidate site:

1. attention is averaged across the four attention heads;
2. self-loop edges are excluded;
3. the maximum outgoing-edge attention is retained for each source-node
   position; and
4. positions from -50 to +50 relative to the candidate adenosine are retained,
   producing 101 positional attention features.

The Baseline Combined GAT remains frozen throughout.

### Supervised attention probe: panels B-D

1. Extract layer-2 attention from the complete validation split
   (3,793 sites from 139 duplexes).
2. Fit a 200-tree XGBoost classifier to all 101 positional features using
   validation labels.
3. Calculate SHAP values on validation and rank positions by mean absolute
   SHAP value.
4. Fit a second XGBoost classifier using the 20 highest-ranked validation
   positions.
5. Evaluate the frozen GAT and both XGBoost probes on the complete held-out
   test split (4,864 sites from 179 duplexes).

Validation and test share no duplexes. The test split is not used for
XGBoost fitting, SHAP ranking, or top-20 feature selection. The GAT validation
split was used during model development to select the GAT checkpoint and its
classification threshold, but not for GAT gradient training.

The GAT uses its validation-selected threshold of 0.475. Both XGBoost probes
use the fixed probability threshold 0.5.

### Descriptive attention profiles: panels E-G

The descriptive panels use all 4,864 held-out test sites without filtering
according to model correctness:

- panel E: mean attention by edited versus not-edited label;
- panel F: mean attention by nucleotide identity; and
- panel G: mean attention by paired versus unpaired structural status.

This population contains 2,539 edited and 2,325 not-edited sites, comprising
470,530 node-position rows. Exact population sizes are recorded in
`data/descriptive_panel_population.json`.

### Positional-availability sensitivity analysis

Positions outside the available sequence window are represented as zero in the
101-position attention matrix. Two controls test whether this boundary pattern
accounts for the attention-probe result:

1. an availability-only XGBoost classifier receives 101 binary indicators of
   whether each relative position exists; and
2. the attention probe is repeated using only sites containing every position
   from -50 to +50, eliminating positional missingness from the analysis.

Both control classifiers are fitted on validation only and evaluated on the
duplex-disjoint test split. SHAP ranking is likewise confined to validation.
The control requires XGBoost 1.7.5, matching the reported attention probe.

## Panel definitions

- **B:** accuracy, precision, recall, and F1 on the same complete test split
  for the GAT, the 101-position XGBoost probe, and the top-20 probe.
- **C:** validation-set SHAP summary for the 101-position probe.
- **D:** validation-set SHAP summary for the top-20 probe.
- **E:** edited and not-edited layer-2 attention profiles across all test
  sites.
- **F:** nucleotide-stratified layer-2 attention profiles across all test
  sites.
- **G:** paired/unpaired layer-2 attention profiles across all test sites.

Attention and SHAP identify learned associations. They do not establish
causal nucleotide effects or a complete mechanistic explanation of the GAT.

## Reproduce the analysis

From the repository root, create the dedicated attention-analysis environment
once:

```bash
conda env create \
  -f analyses/attention_interpretability/environment.yml
conda activate adaredit-attention
```

This environment pins XGBoost 1.7.5, SHAP 0.49.1, NumPy 1.26.4,
scikit-learn 1.6.1, and the complete plotting stack. PyTorch 1.11.0 and
PyTorch Geometric 2.0.4 are pinned because attention coefficients can differ
slightly between PyG implementations even when checkpoint predictions remain
numerically equivalent. These versions reproduce the supplied positional
attention values, validation-derived SHAP ranking, and probe metrics.
`environment_versions.json` records the complete software provenance.

Do not use the repository-level environment for this analysis: `run_all.sh`
checks every relevant version before reading the model or fitting XGBoost and
fails with a detailed message if the dedicated environment is not active.

Then run:

```bash
cd analyses/attention_interpretability
bash run_all.sh
```

To select a Python executable or attention-extraction device:

```bash
PYTHON=/path/to/python DEVICE=cpu bash run_all.sh
PYTHON=/path/to/python DEVICE=cuda bash run_all.sh
```

CUDA is optional. XGBoost uses CPU histogram training with eight threads.
All paths are resolved relative to this directory; no institutional cluster,
private server, or user-specific path is required.

## Directory contents

```text
attention_interpretability/
├── README.md
├── RESULTS.md
├── environment.yml
├── environment_versions.json
├── reference_results.json
├── run_all.sh
├── checkpoint/
│   ├── best.pth
│   ├── summary.json
│   └── test_predictions.csv
├── code/
│   └── gnnadar_verb_compact.py
├── input/
│   ├── Combined_valid.jsonl
│   └── Combined_test.jsonl
├── scripts/
│   ├── extract_attention.py
│   ├── check_environment.py
│   ├── train_attention_probe.py
│   ├── make_figure.py
│   ├── run_availability_control.py
│   ├── make_availability_figure.py
│   ├── validate_availability_control.py
│   └── validate_outputs.py
├── data/
├── models/
└── figures/
```

Important outputs:

- `figures/attention_interpretability.png`: composite panels B-G;
- `figures/attention_availability_control.png`: supplementary boundary-
  availability sensitivity analysis;
- `figures/panels/`: standalone primary panels in PNG and PDF formats;
- `data/metrics_L2.json`: complete validation and test metrics;
- `data/runtime_environment.json`: required and observed software versions;
- `data/availability_control_metrics.json`: complete sensitivity-analysis
  protocol, counts, software versions, metrics and SHAP ranking;
- `data/predictions_test_L2.csv`: test predictions from all three classifiers;
- `data/top20_validation_features_L2.json`: validation-ranked positions;
- `data/node_level_attention_test_L2.csv`: node-level table behind panels E-G;
  and
- `models/`: serialized XGBoost models.

See `RESULTS.md` for the expected numerical results.

The final validator compares the regenerated test metrics and validation-SHAP
rankings with `reference_results.json` using an absolute metric tolerance of
`5e-4`. It also verifies the split counts, absence of validation/test duplex overlap,
checkpoint-inference fidelity, complete-window sensitivity analysis, and all
required output files. A successful run ends with:

```text
PASS: attention-interpretability analysis completed
```
