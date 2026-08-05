# Training dynamics and validation-selected thresholds

This analysis reproduces the supplementary training-dynamics figure from the
16 per-context models: five human settings and three within-species settings,
for both the Baseline GAT and Bio-aware GNN.

## Run

From the repository root:

```bash
cd analyses/minor_training_dynamics
bash run_all.sh
```

No GPU or model retraining is required. The script reads the training histories
and summaries already included under `checkpoints/`.

Successful completion ends with:

```text
PASS: training-dynamics validation
[training dynamics] PASS
```

## Inputs

For every combination of architecture and setting, the analysis reads:

```text
checkpoints/<architecture>_<setting>/history.csv
checkpoints/<architecture>_<setting>/summary.json
checkpoints/<architecture>_<setting>/run_config.json
```

Architectures:

```text
baseline
bioaware
```

Settings:

```text
Artery
Brain
Liver
MuscleSkeletal
Combined
Octopus
Ptychodera
Strongylocentrotus
```

Before plotting, the workflow verifies that every history:

- contains exactly one row for every epoch from 1 through 1,000;
- contains finite training loss, validation F1 and threshold values;
- records validation-only model selection;
- agrees with the `best_epoch`, best validation F1 and selected threshold in
  its `summary.json`;
- used seed 42 and the 1,000-epoch training budget; and
- records the graph-construction invariants used by the reported models.

## Figure panels

`training_dynamics.png` and `.pdf` contain:

- **a:** training loss, shown with a 51-epoch moving average;
- **b:** validation F1 at the best threshold for each epoch, also smoothed;
- **c:** the validation F1-optimal threshold at every epoch, also smoothed;
- **d:** the running-best validation F1;
- **e:** the final validation-selected threshold for each of the eight
  settings, shown separately for both architectures.

The smoothing is used only for visualizing panels a--c. It does not affect
checkpoint selection, threshold selection, reported metrics or panel e.

## Outputs

```text
analyses/minor_training_dynamics/training_dynamics.png
analyses/minor_training_dynamics/training_dynamics.pdf
analyses/minor_training_dynamics/training_dynamics_summary.csv
analyses/minor_training_dynamics/training_dynamics_summary.json
manuscript/training_dynamics.png
```

The summary files record the best epoch, best validation F1, selected threshold
and SHA-256 provenance for every history and summary used in the figure.

## Validation-selected thresholds

| Setting | Baseline GAT | Bio-aware GNN |
|---|---:|---:|
| Artery Tibial | 0.450 | 0.400 |
| Brain Cerebellum | 0.375 | 0.250 |
| Liver | 0.525 | 0.375 |
| Muscle Skeletal | 0.350 | 0.100 |
| Combined | 0.475 | 0.250 |
| *O. bimaculoides* | 0.600 | 0.325 |
| *P. flava* | 0.475 | 0.100 |
| *S. purpuratus* | 0.400 | 0.375 |

Across the eight settings, the Baseline GAT median is **0.4625** (range
0.350--0.600) and the Bio-aware GNN median is **0.2875** (range
0.100--0.400). These validation-selected thresholds set the operating point
for threshold-dependent metrics such as F1. AUROC and AUPRC do not depend on
the selected threshold.

## Code

- `make_training_dynamics.py` validates the 16 input histories, writes the
  provenance summaries and generates the figure.
- `validate_outputs.py` independently checks the expected thresholds, summary
  statistics and manuscript figure copy.
- `run_all.sh` runs the complete workflow.

All paths are resolved relative to the repository structure; the scripts do
not depend on the name or absolute location of the repository directory.
