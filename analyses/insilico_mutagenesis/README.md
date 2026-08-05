# In-silico mutagenesis of the Baseline Combined model

This directory reproduces all panels of the in-silico mutagenesis figure from
the `baseline_Combined` model. It is self-contained: the exact model
checkpoint, validation input, graph-construction code, analysis scripts,
source tables and rendered figures are stored here.

## Analysis population

The analysis uses the Combined validation split. It retains records with a
positive experimental label and an original model prediction greater than
0.7. The held-out test split is not used for selecting or probing examples.

The model is the homogeneous-edge baseline GAT. Each directed
backbone or base-pair relation occurs exactly once, and both `T` and `U` use
the same nucleotide channel.

## Panels

- **A, sequence preference:** At positions -3, -2, -1, +1, +2 and +3 relative
  to the target A, the focal nucleotide is separately replaced by A, G, C or
  T while the original graph topology is retained. Each cell is the mean
  prediction centered on the four-base mean at that position.
- **B, existing-pair retention by base:** The same focal-base substitutions
  are applied, but only at positions paired in the original RNAfold
  structure. The plotted effect is the prediction with the existing pair
  retained minus the prediction after that exact pair is disrupted.
- **C, paired-state indicator sensitivity across the target window:** At every
  position from -40 to +40, the focal node's paired-state indicator is set
  once to 1 and once to 0. The plotted effect is the prediction with indicator
  1 minus the prediction with indicator 0. Sequence, graph edges and all other
  node features are held fixed, and both originally paired and originally
  unpaired positions are included. The shaded band is the standard error of
  the mean across eligible sites.
- **D, focal/partner interaction:** At -1, 0 and +1, both the focal nucleotide
  and its original structural partner are substituted. Watson-Crick and G-T
  combinations retain the pair; other combinations remove the pair. At
  position 0 the focal nucleotide remains A.

In panels B and D, structural disruption removes both directed pair edges and
clears the paired feature for both partners. Panel C is instead a feature-level
sensitivity analysis: it changes only the focal node's paired-state indicator
and neither creates nor removes a physical pair. The analysis never assigns an
invented partner to an originally unpaired nucleotide. RNAfold is not rerun
after any intervention.

The panels describe model behavior in the selected high-confidence positive
population. They do not establish causal biochemical effects or show that the
model has learned every thermodynamic pairing rule. In particular, panel D
changes both sequence and topology for noncanonical combinations and should be
interpreted as a counterfactual interaction probe.

## Run

Use the repository environment or create an environment containing Python
3.9 or later, PyTorch, PyTorch Geometric, NumPy, Matplotlib and seaborn. The
versions used to render the included outputs are recorded in
`environment_versions.json`. Then run:

```bash
bash run_all.sh
```

To select a device or change the inference batch size:

```bash
DEVICE=cuda BATCH_SIZE=64 bash run_all.sh
```

For a quick smoke test that does not overwrite the complete published
outputs, copy this directory elsewhere and run:

```bash
python scripts/run_mutagenesis.py --device cpu --max-selected 8
python scripts/make_figure.py
```

## Outputs

`data/selected_sites.csv` records the exact analyzed validation examples.
Each panel has a sample-level table and a compact summary table:

- `data/panel_A_sequence_mutagenesis.csv` and `panel_A_summary.csv`
- `data/panel_B_pair_disruption_by_base.csv` and `panel_B_summary.csv`
- `data/panel_C_pairing_indicator_by_position.csv` and `panel_C_summary.csv`
- `data/panel_D_pair_interactions.csv` and `panel_D_summary.csv`

`data/analysis_metadata.json` records the selection rule, hashes, graph
validation checks, checkpoint epoch and run configuration.

The main output is `figures/insilico_mutagenesis_full.png` (also PDF).
Individual panels are provided in both formats.

## Directory contents

- `checkpoint/`: exact Baseline Combined checkpoint and training summary
- `code/`: baseline model and homogeneous graph constructor
- `input/`: Combined validation JSONL
- `scripts/`: analysis and plotting code
- `data/`: numeric source data for every panel
- `figures/`: individual panels and composite figure
