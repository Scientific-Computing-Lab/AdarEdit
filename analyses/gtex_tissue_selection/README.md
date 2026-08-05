# GTEx tissue-selection rationale (Supplementary Fig. S2)

This analysis visualizes median GTEx v8 expression of the two catalytically
active ADAR-family genes, `ADAR` (ADAR1) and `ADARB1` (ADAR2), in the four
individual human tissue contexts used in the manuscript.

The analysis is descriptive and independent of model training, graph
construction, checkpoints, and predictions.

## Input

`data/gtex_v8_adar_expression.csv` is a frozen snapshot of median gene-level
TPM from the GTEx v8 API:

- endpoint: `https://gtexportal.org/api/v2/expression/medianGeneExpression`
- dataset: `gtex_v8`
- `ADAR`: `ENSG00000160710.15`
- `ADARB1`: `ENSG00000197381.15`

The full 54-tissue table is retained so the statements that Artery Tibial has
the highest median `ADARB1` expression and Muscle Skeletal the lowest median
`ADAR` expression can be checked against all GTEx v8 tissues, rather than only
against the four plotted tissues.

## Reproduce

From this directory:

```bash
bash run_all.sh
```

or, from the repository root:

```bash
bash analyses/gtex_tissue_selection/run_all.sh
```

The workflow:

1. validates the frozen GTEx v8 values and the tissue-ranking claims;
2. creates `figures/gtex_tissues.png` and `.pdf`;
3. copies the manuscript-ready PNG to `manuscript/gtex_tissues.png`; and
4. validates all outputs.

The plotting workflow is fully local and requires only Python, pandas,
matplotlib, and NumPy. It does not access the network or contain
machine-specific paths.
