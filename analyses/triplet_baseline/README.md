# Triplet-SVM baseline (is a light structure-string model sufficient?)

## What this shows

This tests whether a simple, light baseline — a modified Triplet-SVM (Xue et
al. 2005, microRNA prediction — k-mers of the Vienna dot-bracket string
together with the base in structural context, e.g. `G(.(` = a bulged/looped
G) — is already sufficient to match the graph attention model. It is not: the
graph models outperform the Triplet-SVM heads by **0.061–0.069 F1** and
**0.123–0.126 AUROC** on the held-out Liver test set.

## What this folder contains

```
triplet_baseline/
├── README.md                         ← this file
├── scripts/
│   ├── train_triplet_baseline.py     ← the Triplet-SVM trainer
│   ├── build_metadata.py    ← adapter: jsonl → {split}.metadata.csv
│   ├── run_all.sh                    ← one-command reproduction (linear heads ~10 s; +RBF, no GPU)
│   └── summarize_triplet_baselines.py← summary helper
└── results/                          ← metrics, predictions and summary.{csv,md}
```

## Reproduce

```
bash scripts/run_all.sh Liver
```

The tissue argument can be replaced with another shipped human context.
`results/summary.csv` and `results/summary.md` index every supplied run; the
manuscript comparison uses the three Liver heads.

## Method (the faithful Triplet-SVM)

For each candidate adenosine, a **target-centered ±25-nt window** over the RNAfold structure is encoded
as **Xue-style triplet features**: for every position the feature is `{base}:{pattern}`, where
`base ∈ {A,C,G,T}` and `pattern` is the paired/unpaired state (`(` = paired, collapsing both `(` and
`)`; `.` = unpaired) of the three-nucleotide local context — the 8 patterns `(((, ((., (.(, (.., .((,
.(., ..(, ...`. This gives 4×8 = **32 triplet features** (normalized counts over the window) plus 4
global features (window fill, fraction paired, target paired?, target relative position) = **36-dim**
feature vector. A `StandardScaler` + `LogisticRegression` / `LinearSVC` / RBF-kernel `SVC` is fit; the decision threshold
is chosen on validation by F1 (grid 0.1–0.9), and the C hyper-parameter by validation F1
(grid {0.1, 1, 10}). This is the Triplet-SVM (base + local dot-bracket triplet). It is a **light model on 36 features** — the linear heads train in ~7 s on CPU and the
RBF variant in a few minutes (no GPU, no epochs), which is the whole point of the "light
baseline" test.

## Results (Liver, held-out test, n = 4,150)

| Model | F1 | AUROC | precision | recall | specificity | source |
|---|---:|---:|---:|---:|---:|---|
| Triplet logistic regression | **0.785** | **0.784** | 0.714 | 0.872 | 0.546 | `results/Liver_logreg.json` |
| Triplet linear SVM | **0.785** | **0.784** | 0.717 | 0.867 | 0.554 | `results/Liver_linear_svm.json` |
| Triplet **RBF-kernel SVM** | **0.793** | **0.787** | 0.699 | 0.915 | 0.489 | `results/Liver_rbf_svm.json` |
| Baseline GNN | **0.854** | **0.909** | — | — | — | trained GNN model (not retrained here) |
| Bio-aware GNN | **0.854** | **0.910** | — | — | — | trained GNN model (not retrained here) |

The graph models exceed the three Triplet heads by 0.061–0.069 F1 and
0.123–0.126 AUROC. A simple structure-string model is therefore not sufficient
to match AdarEdit; the AUROC difference shows substantially stronger ranking by
the graph models.

**The result does not depend on the classifier head.** Xue et al. (2005) use an RBF-kernel SVM
(LibSVM, with `C` and `gamma` tuned by grid search), so that variant is run here too. It performs equivalently to the two linear heads (F1 0.793 vs 0.785; AUROC 0.787 vs 0.784),
confirming that the graph model's margin is not an artefact of pairing a graph network against a
deliberately linear control. All three heads are seeded (`SEED = 42`), so the numbers reproduce
run-to-run; `SVC(probability=True)` fits Platt scaling by internal CV and is not deterministic
without it.

**Reading the numbers.** Liver's held-out test set is 56.5 % positive, so predicting "edited"
everywhere already yields F1 = 0.722; on F1 alone the Triplet control clears that floor by only
~0.06. AUROC is the informative comparison here: it is threshold-free and a trivial predictor scores
exactly 0.5, so the control's 0.78 shows it genuinely ranks edited above non-edited sites — and the
graph models' 0.909–0.910 are a further 0.12–0.13 above it. The control's low specificity (~0.55 at its
validation-selected threshold) reflects the same permissive operating point, not a lack of signal.

## Notes

- The GNN numbers are the already-trained models; only the Triplet baselines are trained here.
  Both use the same global pair-disjoint Liver split (same 4,150 held-out sites).
- All shipped Liver records satisfy `len(full_seq) == len(structure)`:
  train 12,485, validation 3,024 and test 4,150.
