# Attention-interpretability results

## Data separation

| role | GAT split | sites | duplexes |
|---|---|---:|---:|
| XGBoost fitting and SHAP feature selection | validation | 3,793 | 139 |
| final performance evaluation | test | 4,864 | 179 |

The validation and test splits have zero duplex overlap. The test split is not
used for XGBoost fitting or feature selection.

## Complete test results

All three classifiers are evaluated on the same 4,864 test sites.

| classifier | accuracy | precision | recall | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Combined GAT | 0.8557 | 0.8334 | 0.9043 | 0.8674 | 0.9331 | 0.9356 |
| XGBoost, all 101 attention positions | 0.7956 | 0.8074 | 0.7991 | 0.8032 | 0.8772 | 0.8790 |
| XGBoost, top 20 validation-ranked positions | 0.7837 | 0.7887 | 0.7999 | 0.7943 | 0.8654 | 0.8636 |

The attention-only probe therefore contains generalizable class information,
although it does not match the complete GAT, which also uses learned node
representations and graph-level pooling.

Both XGBoost probes fit the validation data nearly perfectly (F1 1.0000 and
0.9997). Interpretation must therefore rely on their held-out test results,
not their validation fitting scores.

## Validation-ranked positions

The 20 positions selected by validation mean absolute SHAP value are:

```text
+1, 0, -2, -1, +22, +2, +12, +13, +24, +20,
+21, +11, -3, +19, +14, +33, +16, +15, -18, +25
```

The four leading positions are +1, 0, -2, and -1.

## Positional-availability sensitivity analysis

Of the 3,793 validation sites, 3,150 (83.0%) contained the complete -50 to +50
window; 4,077 of 4,864 test sites (83.8%) met the same criterion. Positions
+1, 0, -2 and -1 were available in every validation and test example.

An XGBoost classifier receiving only binary indicators of positional
availability showed weak held-out discrimination (AUROC 0.574, AUPRC 0.565).
The original all-position attention probe achieved F1 0.803, AUROC 0.877 and
AUPRC 0.879 on the full test split. When evaluated only on complete-window test
sites, the same fitted probe retained F1 0.801, AUROC 0.856 and AUPRC 0.877.
Refitting the identical XGBoost specification on complete-window validation
sites yielded F1 0.807, AUROC 0.856 and AUPRC 0.879 on the complete-window test
subset.

The complete-window validation SHAP ranking began with +1, 0, -2, +22, +12,
+13 and -1. Thus, the three leading proximal positions were unchanged and all
four proximal positions remained among the seven highest-ranked features after
positional missingness was removed. Sequence-boundary availability carries a
weak label-associated signal, but it does not account for the held-out
attention-probe performance or the dominant proximal ranking.

## Descriptive test-set profiles

Panels E-G summarize all 4,864 test sites without filtering according to
model correctness: 2,539 edited and 2,325 not-edited sites. They contain
470,530 node-position rows spanning -50 to +50.

At position -1, mean attention is highest for G (0.515), followed by U
(0.472), C (0.465), and A (0.404). Across the complete window, mean attention
is higher at unpaired positions (0.366) than at paired positions (0.304).

## Reproducibility checks

- Validation/test duplex overlap: 0.
- Test examples used for XGBoost fitting or feature selection: no.
- GAT test probabilities reproduce the checkpoint prediction table with a
  maximum absolute difference of \(1.32\times10^{-5}\).
- Reloaded XGBoost models reproduce their stored test probabilities with a
  maximum absolute difference below \(3.0\times10^{-8}\).
- Positional feature range: -50 to +50 inclusive.
- Random seed: 42.

These results show that positional attention contains predictive information
and highlights proximal positions. Attention and SHAP remain associative
interpretability tools and do not establish causal biochemical mechanisms.
