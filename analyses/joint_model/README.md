# Joint multi-tissue models

This analysis trains and evaluates one multi-label model across the human
tissue contexts. A shared encoder produces one editing probability per tissue.
If a site has no label for a particular tissue, that output is excluded from
the binary-cross-entropy loss with an observation mask.

Four models are reported:

| Encoder | Outputs | Run name |
|---|---|---|
| Baseline | Artery, Brain, Combined, Liver, Muscle Skeletal | `joint_baseline` |
| Bio-aware | Artery, Brain, Combined, Liver, Muscle Skeletal | `joint_bioaware` |
| Baseline | Artery, Brain, Liver, Muscle Skeletal | `joint_baseline_noCombined` |
| Bio-aware | Artery, Brain, Liver, Muscle Skeletal | `joint_bioaware_noCombined` |

The two no-Combined models are a supervision control. They test whether
including the aggregate Combined label as a fifth output is responsible for
the performance of the joint model on the four tissue-specific outputs.

## Model and graph construction

`pipeline/train_joint.py` is the trainer used for all four runs. The encoder is
selected with `--variant baseline|bioaware`; `--tissue-set
with_combined|no_combined` selects five or four output nodes.

The bundled graph constructors implement the graph representation used by the
reported models:

- every directed backbone or base-pair relation occurs exactly once;
- `T` and `U` share the same nucleotide channel;
- the bio-aware stem feature is the full contiguous helix length;
- pair compatibility is zero on backbone edges;
- canonical, wobble and other structural pair edges have distinct edge types
  in the bio-aware graph.

The Baseline model has an eight-feature node representation and untyped edges.
The Bio-aware model adds neighboring-base, structural and edge annotations and
a sequence-CNN branch. Both joint models use a masked multi-label loss.

## Data and split integrity

Inputs are the shared repository files under `../../data/human/`. Whole
substrate pairs are assigned to train, validation or test once and the same
assignment is reused across tissues. `scripts/check_joint_leakage.py` verifies
both `pair_id` disjointness from the metadata and site disjointness from the
JSONL records before training.

The with-Combined dataset contains 37,014 unique training sites, 9,119
validation sites and 11,285 test sites. Because not every site is observed in
every tissue, these correspond to 63,388, 15,747 and 19,797 observed labels,
respectively. Excluding Combined gives 31,391, 7,799 and 9,596 unique sites and
48,083, 11,954 and 14,933 observed tissue labels.

Model selection is based only on macro F1 across the observed validation
labels. A separate F1-optimal threshold is selected on validation for each
output and then applied unchanged to test. AUROC and AUPRC are threshold-free.
The test split is not evaluated during training or checkpoint selection.

## Supplied outputs

The exact best checkpoints, histories, prediction files and summaries are in:

```text
../../checkpoints/joint_baseline/
../../checkpoints/joint_bioaware/
../../checkpoints/joint_baseline_noCombined/
../../checkpoints/joint_bioaware_noCombined/
```

Copies of the four summary files are retained under `results/`. The reported
best epochs and held-out macro metrics are:

| Run | Best epoch | Macro F1 | Macro AUROC | Macro AUPRC |
|---|---:|---:|---:|---:|
| Joint Baseline | 704 | 0.851 | 0.911 | 0.915 |
| Joint Bio-aware | 875 | 0.855 | 0.917 | 0.929 |
| Joint Baseline, no Combined | 225 | 0.858 | 0.914 | 0.919 |
| Joint Bio-aware, no Combined | 136 | 0.856 | 0.920 | 0.927 |

The no-Combined and with-Combined macro values are not directly comparable when
the former has four outputs and the latter has five. On the same four
tissue-specific test outputs, removing Combined changes:

- Baseline macro F1 from 0.849 to 0.858 and macro AUROC from 0.909 to 0.914.
- Bio-aware macro F1 from 0.855 to 0.856 and macro AUROC from 0.916 to 0.920.

At the reported seed, tissue-specific performance is retained when Combined is
removed; the observed joint performance therefore cannot be attributed solely
to supervision from the aggregate Combined label. This is a single-seed
control, not a variance estimate across repeated training runs. It also does
not establish that every tissue benefits equally from joint training; that
comparison is provided in `../joint_vs_pertissue/`.

## Reproduce the supplied metrics and control

Create and activate the repository environment first:

```bash
conda env create -f ../../environment.yml
conda activate adaredit
```

The supplied checkpoints were produced with PyTorch 2.7.0 and PyTorch
Geometric 2.6.1; these versions are pinned in `../../environment.yml`.
PyTorch Geometric 2.0.x is not checkpoint-compatible with these runs because
the internal `GATConv` state-dictionary keys differ.

Then, from this directory:

```bash
bash run_all.sh
```

This CPU-compatible command:

1. verifies pair- and site-level split disjointness;
2. recomputes all per-tissue, macro and pooled metrics from the supplied test
   predictions and checks them against the four summaries;
3. writes `results/combined_supervision_control.csv`;
4. renders `figures/combined_supervision_control.{png,pdf}` and the
   manuscript-ready copy `../../manuscript/figS_joint_combined_control.png`.

The control figure contains four panels (Baseline/Bio-aware by F1/AUROC).
Within each tissue, an open marker denotes the five-output model trained with
Combined and a filled marker denotes the four-output model trained without
Combined; the two values are connected by a line. All four panels use the same
absolute score range (0.80--0.95) so small single-seed differences are not
visually exaggerated by truncated, data-adaptive axes.

## Retrain

A CUDA GPU is recommended for the complete 1,000-epoch runs:

```bash
bash scripts/run_joint.sh
```

By default this trains all four models sequentially and writes new outputs
under `outputs/`, leaving the supplied checkpoints unchanged. To run one model:

```bash
RUNS="bioaware:no_combined" bash scripts/run_joint.sh
```

Useful environment variables are documented in `scripts/config.env`, including
`EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`, `NUM_THREADS`, `OUT_ROOT` and
`CACHE_ROOT`. For a CPU smoke test only:

```bash
ALLOW_CPU=1 EPOCHS=1 RUNS="baseline:no_combined" bash scripts/run_joint.sh
```
