# Substrate structural stability — full analysis documentation

This analysis (1) characterises how structurally stable the edited *Alu* duplexes are — both
intrinsically and in their genomic environment — and (2) shows that the model's predictive
performance holds for the less-stable substrates rather than being driven only by the most rigid
hairpins.

This document records exactly what is computed, from which files, with which method, and how many
substrates/sites enter each analysis. All three analyses feed a single consolidated figure
(`figures/substrate_stability.png`, four panels a–d).

Everything is run on the **global pair-disjoint split**, in which whole *Alu* pairs — never
individual sites — are assigned to disjoint train/validation/test sets (64:16:20), so a substrate
seen at test time was never seen during training. All performance numbers below are on the
**held-out test split only**.

--------------------------------------------------------------------------------------------------

## Analysis 1 — Intrinsic substrate stability (Figure panel a)

**Question.** How stable is the minimum-free-energy secondary structure of each *Alu* duplex, as
the model actually sees it (the dot-bracket string provided as input)?

**Stability metric.** *Paired fraction* = (number of `(` + number of `)`) / (length of the
dot-bracket structure). This is the fraction of nucleotides engaged in a base pair — 1.0 = fully
double-stranded, 0.0 = fully single-stranded.

**Input files.**
- `data/human/Combined/train.jsonl`
- `.../Combined/valid.jsonl`
- `.../Combined/test.jsonl`
  (the *Combined* setting contains every human *Alu* substrate; the structure is
  tissue-independent, so this is the complete substrate set.)

**Method.** For each record we parse the `Structure:` field (the RNAfold dot-bracket the model was
given) and the reconstructed sequence `L + "A" + R`. Records for which
`len(sequence) != len(structure)` are dropped — these are exactly the records the graph builder
itself skips (it cannot map a residue-per-node graph when the two strings disagree). Substrates
are then de-duplicated by their **full** dot-bracket structure (not by a prefix/suffix key), and
the paired fraction is computed once per unique substrate.

**Output / numbers.**
- Unique substrates entering the distribution: **884**
- Paired fraction: min **0.655**, Q1 **0.776**, median **0.812**, Q3 **0.844**, max **0.917**
- The Q1 (0.776) and Q3 (0.844) cutoffs define the low- and high-stability cohorts used in
  panels c–d (see Analysis 3).

**Script.** `scripts/stability_performance.py` (the `_global` block).
**Data written to.** `data/stability_performance.json` → key `_global`.

--------------------------------------------------------------------------------------------------

## Analysis 2 — Hairpin maintenance in genomic context (Figure panel b)

**Question.** A duplex folded in isolation may not survive once it is embedded in its native
genomic neighbourhood. Do the two *Alu* arms still base-pair with each other when the pair is
re-folded together with the natural flanking sequence?

**Metric.** `dsrna_frac` = the fraction of arm-1 nucleotides that pair **specifically with arm 2**
in the MFE structure of the full context construct (i.e. genuine inter-arm duplex, not incidental
local pairing).

**Input files.**
- `raw_data/Pair_Alu_withStrand.bed` — genomic
  coordinates (chr, start1/end1 of arm 1, start2/end2 of arm 2, strand) of each *Alu* pair.
- The **hg38 reference genome** FASTA (external download, e.g. UCSC `hg38.fa`), read via `pyfaidx`.

**Method.** For each pair the construct is built as
`[200 bp flank] + arm1 + [natural inter-Alu genomic sequence] + arm2 + [200 bp flank]`
(FLANK = 200 bp on each side; constructs longer than MAX_SEQ_LEN = 3500 nt are excluded). Each
construct is folded with ViennaRNA including the **partition function** (`RNA.fold` +
`RNA.pf_fold`), and `dsrna_frac`, `delta_delta_g` (G_MFE − G_ensemble) and the Boltzmann
arm1–arm2 pairing probability are recorded. Because this step needs the full hg38 genome, its
result table `raw_data/context_folding_results.csv` is **provided pre-computed**; the downstream
panel and statistics read directly from it. The reported panel uses `dsrna_frac`; the ensemble
metrics are retained in the CSV but not plotted (the per-base-pair Boltzmann average is dominated
by structural zeros and is not informative as a single scalar).

**Output / numbers.**
- Substrates folded in context: **865** (pairs within the 3500 nt length cap)
- `dsrna_frac`: median **0.810**, mean 0.766
- **94.6 %** of substrates retain `dsrna_frac ≥ 0.5` — the inter-arm hairpin is preserved even
  with 200 bp of competing flanking sequence on each side.

**Data (pre-computed).** `raw_data/context_folding_results.csv`

*Note on counts.* Analysis 1 yields 884 substrates (unique input structures) and Analysis 2 yields
865 (pairs that folded within the length cap). The two numbers differ because they are independent
structural computations on slightly different valid subsets (input-structure parsing vs.
context-folding length cap); each is reported transparently against its own denominator.

--------------------------------------------------------------------------------------------------

## Analysis 3 — Predictive performance by stability cohort (Figure panels c, d)

**Question.** Is the model's accuracy driven only by the most stable hairpins, or does it still
discriminate edited from non-edited adenosines on the least stable substrates?

**Cohorts.** Using the **global** paired-fraction quartiles from Analysis 1:
- **Low stability** = sites whose substrate has paired fraction ≤ Q1 (0.776)
- **High stability** = sites whose substrate has paired fraction ≥ Q3 (0.844)
The same absolute thresholds are applied to every tissue, so the cohorts mean the same thing
across settings.

**Input files (per tissue T ∈ {Artery, Brain, Liver, MuscleSkeletal, Combined}).**
- Labels/structures: `.../data/human/{T}/test.jsonl`
- Baseline predictions:
  `checkpoints/baseline_{T}/test_predictions.csv`
- Bio-aware predictions:
  `checkpoints/bioaware_{T}/test_predictions.csv`
  (columns include `pair_id, pair_prefix, yes_no, split, prob, label_from_loader,
  pred_label, threshold, variant, tissue, graph_version`; `threshold` is the
  validation-selected decision threshold for that model.)

**Alignment (collision-free, no prefix keys).** The prediction CSV is emitted in the same order as
the graph builder consumed the test set. Every shipped test record satisfies
`len(sequence) == len(structure)`, so the JSONL records and prediction rows align
**1:1 by position**. The
alignment is verified for every tissue×architecture by asserting that
`label_from_loader == the jsonl label` at every position (`assert` in the script; all pass). Each
prediction is thus tied to its **exact** substrate and full structure — no start/end prefix
matching, no collisions.

Alignment counts per tissue (test split):

| Tissue          | test records | aligned | skipped |
|-----------------|-------------:|--------:|--------:|
| Artery          | 4792         | 4792    | 0       |
| Brain           | 4566         | 4566    | 0       |
| Liver           | 4150         | 4150    | 0       |
| Muscle Skeletal | 1425         | 1425    | 0       |
| Combined        | 4864         | 4864    | 0       |

**Metrics.** For each cohort we compute F1 at the model's own validation-selected threshold
(the `threshold` column) and AUROC (threshold-free, rank-based). AUROC is the primary read-out
because it is independent of the operating threshold.

**Output / numbers (held-out test).**

AUROC — low (≤Q1) vs high (≥Q3) stability cohort:

| Tissue   | Baseline low | Baseline high | Bio-aware low | Bio-aware high |
|----------|-------------:|--------------:|--------------:|---------------:|
| Artery   | 0.896        | 0.946         | 0.887         | 0.932          |
| Brain    | 0.883        | 0.938         | 0.881         | 0.927          |
| Liver    | 0.898        | 0.903         | 0.880         | 0.911          |
| Muscle   | 0.839        | 0.889         | 0.826         | 0.923          |
| Combined | 0.900        | 0.947         | 0.875         | 0.917          |

F1 — low (≤Q1) vs high (≥Q3) stability cohort:

| Tissue   | Baseline low | Baseline high | Bio-aware low | Bio-aware high |
|----------|-------------:|--------------:|--------------:|---------------:|
| Artery   | 0.811        | 0.897         | 0.796         | 0.883          |
| Brain    | 0.777        | 0.880         | 0.788         | 0.863          |
| Liver    | 0.790        | 0.876         | 0.784         | 0.869          |
| Muscle   | 0.750        | 0.838         | 0.728         | 0.852          |
| Combined | 0.805        | 0.893         | 0.776         | 0.859          |

Cohort sizes (sites) are given in `data/stability_performance.json`; e.g. Combined
baseline: low n=1283, high n=1354, all n=4864.

**Interpretation.** Discrimination remains strong on the
least-stable quartile — AUROC ≈ 0.88–0.90 in most contexts — and generally rises in the
most-stable quartile. The two architectures therefore retain useful discrimination beyond the
most rigid hairpins. Muscle Skeletal, the smallest and lowest-signal setting, remains weaker in
the low-stability cohort (AUROC 0.826–0.839), consistent with its lower overall performance.

**Script.** `scripts/stability_performance.py`
**Data written to.** `data/stability_performance.json`

--------------------------------------------------------------------------------------------------

## Figure assembly

**Script.** `scripts/make_stability_figure.py`
**Reads.** `data/stability_performance.json` (panels a, c, d) and
`raw_data/context_folding_results.csv` (panel b).
**Writes.** `figures/substrate_stability.png` and `.pdf`.

Panels: (a) intrinsic paired-fraction distribution with Q1/median/Q3 marked; (b) genomic-context
`dsrna_frac` distribution with the 0.5 reference and the 94.6 % annotation; (c) AUROC and (d) F1
for the low- vs high-stability cohorts, both architectures, all five human settings.

## Provenance / reproduction

```bash
# Recompute Analyses 1 and 3, recreate the four-panel figure, copy the
# manuscript-ready PNG, and validate all outputs:
bash run_all.sh

# Optional free-energy summary statistics:
python3 scripts/compute_free_energy_stats.py
```

Analysis 2 uses the provided `raw_data/context_folding_results.csv`. Recomputing
that upstream table requires the full hg38 genome and ViennaRNA, as documented
above. The plotted performance values are read from the shipped per-context test
prediction CSVs, and their SHA-256 hashes are recorded in
`data/stability_performance.json`.
