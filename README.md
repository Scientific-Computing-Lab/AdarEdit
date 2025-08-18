# AdarEdit: A Graph Foundation Model for Interpretable A-to-I RNA Editing Prediction

AdarEdit is a domain-specialized graph foundation model for predicting A-to-I RNA editing sites. Unlike generic foundation models that treat RNA as linear sequences, AdarEdit represents RNA segments as graphs where nucleotides are nodes connected by both sequential and base-pairing edges, enabling the model to learn biologically meaningful sequence–structure patterns.

## Key Features:
- Graph-based RNA representation: Captures both sequence and secondary structure information.
- High accuracy: F1 > 0.85 across cross-tissue evaluations (see Results).
- Cross-species generalization: Works on evolutionarily distant species even without Alu elements.
- Mechanistic interpretability: Graph attention highlights influential structural motifs.
- Foundation model behavior: A single model generalizes across tissues and conditions.

