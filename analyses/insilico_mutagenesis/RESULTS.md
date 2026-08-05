# Results

The complete analysis passed all graph and inference checks. It analyzed all
1,520 validation examples with a positive experimental label and an original
Baseline Combined prediction greater than 0.7. No optional sample cap was
applied.

## Panel A: sequence substitutions

The dominant sequence effect was at position -1. Substitution with G lowered
the position-centered prediction by 0.425, whereas A, C and T increased it by
0.152, 0.122 and 0.151, respectively. A smaller G penalty occurred at -2
(-0.093). At +1, T was disfavored (-0.054), while the other substitutions had
small positive effects (0.008 to 0.023). Effects at +/-3 were close to zero.

## Panel B: disruption of existing pairs

Panel B includes only examples in which the queried position has an RNAfold
partner. Retaining that pair rather than disrupting it had the largest
positive mean effects for C at +1 (0.276; n=1,415) and T at +1 (0.196;
n=1,415). Under a G substitution at -1, pair retention instead lowered the
mean score by 0.302 (n=1,235).

## Panel C: paired-state indicator sensitivity

Panel C changes only the paired-state indicator at the focal node while
holding sequence, graph edges and every other node feature fixed. It includes
both originally paired and originally unpaired positions. Setting this
indicator to 1 rather than 0 increased the mean prediction most strongly at
-1 (0.035), +1 (0.022), -2 (0.020) and +2 (0.010; n=1,520 at each position).
At the target position, the effect was negative (-0.027; n=1,520), indicating
that the model assigns a higher score to the unpaired-state indicator there.
Effects were smaller away from the target and became mildly negative across
much of the far-downstream region.

This is a feature-sensitivity probe, not a physical refolding experiment:
panel C neither creates nor removes a base-pair edge and does not change the
partner node. Its effect therefore measures the model's use of the focal
paired-state feature conditional on a fixed graph, whereas panel B measures
the effect of retaining versus disrupting an existing pair.

## Panel D: focal/partner substitutions

At -1, focal G produced lower predictions across all partner identities
(0.195 to 0.631) than focal A, C or U (predominantly 0.77 to 0.97). At +1,
several retained Watson-Crick/wobble combinations scored highly, including
G-C (0.960), G-T (0.931) and C-G (0.933). The target-A row ranged from 0.835
to 0.976 across partner substitutions.

Panel D is descriptive of the model's counterfactual responses. Because
noncanonical substitutions remove the pair while canonical/wobble
substitutions retain it, sequence and topology change together. The panel
therefore does not by itself establish biochemical pair compatibility.

The authoritative machine-readable record is
`data/analysis_metadata.json`. Exact sample-level values and aggregate means,
standard errors and sample sizes are in the panel CSV files.
