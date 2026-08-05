# Model evaluation results

This directory contains the held-out evaluations used in the manuscript.

## Canonical inputs

`preds/` contains one compressed prediction archive for every combination of:

- architecture: Baseline GAT or Bio-aware GNN;
- training context: five human contexts and three species;
- evaluation context: the same eight contexts.

Each archive stores the prediction scores, labels, validation-selected
threshold, training context, evaluation context, architecture and graph
version. The threshold selected for the training context is applied unchanged
to every evaluation context.

The `matrix_<variant>_<metric>.csv` files provide the same 8 × 8 evaluations in
matrix form for F1, precision, recall, AUROC, AUPRC and threshold.

## Rebuild and validate the tabular results

From the repository root:

```bash
python results/build_results_tables.py
```

The script recomputes every metric directly from the 128 prediction archives
and fails if any value differs from its corresponding matrix entry by more than
`1e-9`. It writes:

- `all_evaluations.csv`: long-format table of all 128 train/evaluation
  combinations;
- `table1_within_context.csv`: the 16 within-context rows, covering both
  architectures across the five human contexts and three species.

The joint multi-tissue results are stored separately under
`checkpoints/joint_*` and documented in `analyses/joint_model/`.
