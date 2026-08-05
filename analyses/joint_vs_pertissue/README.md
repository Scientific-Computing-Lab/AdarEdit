# Joint versus per-tissue models

This analysis compares a single five-output joint model with five independently
trained per-tissue models. Baseline is compared with Baseline and Bio-aware is
compared with Bio-aware, so joint versus per-tissue training is the only
difference within each comparison.

## Models

- A per-tissue model has one encoder and one binary output for a single tissue.
- A joint model has one shared encoder and five outputs: Artery, Brain,
  Combined, Liver and Muscle Skeletal.
- Missing tissue/site labels are masked from the joint training loss.
- Each model is selected on validation F1. Test F1 uses the corresponding
  validation-selected threshold; AUROC is threshold-free.
- All models use the same global pair-disjoint split and are evaluated on the
  same held-out examples for each tissue.

The no-Combined supervision control is reported separately in
`../joint_model/`; the comparison here uses the five-output joint models.

## Source data

`scripts/build_comparison_data.py` constructs `data/comparison_data.csv`
directly from the supplied outputs:

- per-tissue probabilities and labels:
  `../../results/preds/<variant>__train-<tissue>__eval-<tissue>.npz`;
- per-tissue validation-selected thresholds:
  `../../checkpoints/<variant>_<tissue>/summary.json`;
- joint per-tissue test metrics:
  `../../checkpoints/joint_<variant>/summary.json`.

The script also refreshes the two joint summary copies in `model_summaries/`.
No plotted number is entered manually.

The generated CSV contains 20 rows: five tissues, two encoder variants, and
per-tissue versus joint training for each variant.

## Run

From this directory:

```bash
bash run_all.sh
```

This regenerates:

```text
data/comparison_data.csv
model_summaries/joint_baseline_summary.json
model_summaries/joint_bioaware_summary.json
figures/joint_comparison_slope.png
figures/joint_comparison_slope.pdf
figures/joint_comparison_stacked.png
figures/joint_comparison_stacked.pdf
```

The slope plot is the primary view. Open markers are independently trained
per-tissue models and filled markers are joint models. The grouped-bar figure
contains the same values as an alternative presentation.

## Results

Macro-averaging the five tissue rows gives:

| Encoder | Per-tissue F1 | Joint F1 | Difference | Per-tissue AUROC | Joint AUROC | Difference |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.854 | 0.851 | -0.004 | 0.913 | 0.911 | -0.002 |
| Bio-aware | 0.846 | 0.855 | +0.009 | 0.907 | 0.917 | +0.010 |

The effects are not uniform across tissues:

- Joint Baseline improves F1 and AUROC in Liver and Muscle Skeletal, but is
  lower in Artery, Brain and Combined.
- Joint Bio-aware improves both metrics in Artery, Combined, Liver and Muscle
  Skeletal. In Brain it is effectively tied in F1 (-0.0002) and slightly lower
  in AUROC (-0.0016).
- The largest Bio-aware gains are in Muscle Skeletal (F1 +0.020; AUROC +0.019)
  and Combined (F1 +0.018; AUROC +0.021).

These results support a context-dependent benefit from shared multi-tissue
training, particularly for the Bio-aware encoder and the smaller Muscle
Skeletal dataset. They do not support the stronger claim that every joint
model outperforms its per-tissue counterpart in every tissue. The comparison
is descriptive; no paired significance test is included in this analysis.
