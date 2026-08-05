# Coding-target generalization analysis

This folder reproduces the out-of-domain evaluation of the trained RNA-editing
models on seven coding-region targets. The models were trained only on human
inverted-Alu substrates. They are applied here without retraining,
fine-tuning, or coding-target-based model selection.

## Analysis design

- Targets: `AJUBA`, `BLCAP`, `FLNA`, `TTYH2`, `NEIL1`, `GRIA2`, and `GRIA3`.
- Tissue contexts: Brain Cerebellum, Combined, and Liver.
- Models: per-tissue Baseline and Bio-aware models, plus Joint Baseline and
  Joint Bio-aware models.
- Positive class: measured editing level greater than or equal to 15%.
- Negative class: measured editing level below 1%.
- Excluded: editing levels from 1% to below 15%, and any adenosine without a
  position-matched measurement.
- Primary metric: per-gene AUROC. AUROC is reported only when both classes
  contain at least two sites.

The main figure compares the two joint models. Predictions from the two
per-tissue models are also regenerated and included in the extended figure and
results table.

## Reproduce the analysis

Create the repository environment described in the top-level
`environment.yml`, activate it, and run:

```bash
bash analyses/coding_targets/run_all.sh
```

For this standalone folder, run:

```bash
bash run_all.sh
```

The script automatically uses a GPU when one is available and otherwise runs
on CPU. To force a device:

```bash
DEVICE=cpu bash run_all.sh
DEVICE=cuda bash run_all.sh
```

No path inside any script is specific to a workstation or compute cluster.

## Workflow

1. `scripts/generate_predictions.py`
   - loads the supplied best-validation checkpoints;
   - reconstructs each graph using the same model and graph-building code used
     in training;
   - scores every adenosine in each coding target;
   - writes 84 prediction files: four model variants × seven genes × three
     tissue contexts.
2. `scripts/build_summary.py`
   - applies the fixed editing-level labels;
   - excludes the intermediate band;
   - computes threshold-independent AUROC and class counts.
3. `scripts/make_figures.py`
   - creates the joint-model AUROC panel;
   - creates the per-tissue-model AUROC panel;
   - creates an extended four-model panel.

## Inputs

- `data/<gene>/seq*.txt`: sequence and supplied Vienna dot-bracket structure.
- `data/<gene>/dsRNA_structure_with_*_editing_sites_andA20.csv`: measured
  tissue-context editing levels. One BLCAP source row has an out-of-range local
  position and therefore does not map to the first sequence adenosine; that
  adenosine is retained in the raw predictions but excluded from AUROC because
  it has no mapped measurement.
- `checkpoints/`: the eight trained checkpoints used in this analysis, with
  their run configurations and validation/test summaries.
- `code/`: graph builders and model definitions used during training.

The sequence is normalized from T to U before graph construction, matching
training. The baseline graph contains one copy of each directed backbone or
base-pair edge. The bio-aware graph uses true contiguous stem lengths, zero
pair-compatibility values on backbone edges, and a distinct type for
noncanonical structural pairs.

## Outputs

- `predictions/<variant>/<gene>/adenosine_prediction_<tissue>.csv`: raw
  per-adenosine scores and editing levels.
- `results/auroc_summary.csv` and `.json`: all 84 model/gene/tissue results.
- `figures/coding_target_auroc_joint.png` and `.pdf`: primary AUROC panel.
- `figures/coding_target_auroc_per_tissue.png` and `.pdf`: Per-tissue
  Bio-aware and Baseline models.
- `figures/coding_target_auroc_all_models.png` and `.pdf`: all four models.

Sample-size labels use `(n_not-edited,n_edited)`. The dashed horizontal line is
AUROC = 0.5. `n/a` denotes fewer than two sites in at least one class.

## Interpretation

This is a deliberately small, target-specific scope test rather than a broad
coding-site benchmark. AUROC assesses whether edited adenosines tend to receive
higher scores than non-edited adenosines, independent of model decision
thresholds. Sparse targets should therefore be interpreted descriptively and
not as precise per-gene performance estimates.
