# AdarEdit: A Graph Foundation Model for Interpretable A-to-I RNA Editing Prediction

AdarEdit is a domain-specialized graph foundation model for predicting A-to-I RNA editing sites. Unlike generic foundation models that treat RNA as linear sequences, AdarEdit represents RNA segments as graphs where nucleotides are nodes connected by both sequential and base-pairing edges, enabling the model to learn biologically meaningful sequence–structure patterns.

## Key Features:
- Graph-based RNA representation: Captures both sequence and secondary structure information.
- High accuracy: F1 > 0.85 across cross-tissue evaluations (see Results).
- Cross-species generalization: Works on evolutionarily distant species even without Alu elements.
- Mechanistic interpretability: Graph attention highlights influential structural motifs.
- Foundation model behavior: A single model generalizes across tissues and conditions.


![GNN_model](Figure/GNN_model.png)
Figure 1: ADAREDIT model architecture showing RNA-to-graph conversion and Graph Attention Network processing


## Getting Started
### Requirments

First, clone this repository. 
```
git clone https://github.com/Scientific-Computing-Lab/AdarEdit.git
cd AdarEdit
```

You may use the file  `environment.yml` to create anaconda environment with the required packages.

### Steps to Use the environment.yml File:
#### Create the Environment:
1. Save the `environment.yml` file in your project directory, then run the following command:
   
```
conda env create -f environment.yml
```

2. Activate the Environment:
   
```
conda activate rnagnn
```

## Data Processing Pipeline
### Step 1: Human Alu Dataset Construction
The Scripts/Data_preparation/Classification_Data_Creation.py script creates classification datasets for each tissue:

Process:

- Read Alu pair regions from BED file (chr1,start1,end1,chr2,start2,end2,strand)
- Extract RNA sequences for each Alu pair using genome FASTA
- Connect Alu pairs with "NNNNNNNNNN" linker sequence
- Predict secondary structure using ViennaRNA fold_compound
- Extract editing levels from tissue-specific editing files
- Filter sites with >100 read coverage
- Generate full context sequences with structural annotations

Input:

`--pair_region`: BED file with Alu pair coordinates 
`--genome`: Human genome FASTA file
`--editing_site_plus/minus`: Editing level files 
`--editing_level`: Minimum editing threshold (e.g., 10.0)

Output:

data_for_prepare_classification.csv → data/processed/tissues/{tissue}/

```
for tissue in Brain_Cerebellum Artery_Tibial Liver Muscle_Skeletal; do
    python scripts/Classification_Data_Creation_Liver.py \
        --pair_region data/raw/alu_pairs.bed \
        --genome data/raw/hg38.fa \
        --editing_site_plus data/raw/${tissue}_editing_plus.tsv \
        --editing_site_minus data/raw/${tissue}_editing_minus.tsv \
        --editing_level 10.0 \
        --output_dir data/processed/tissues/${tissue}/
done
```

### Step 2: Cross-Tissue Data Splitting
The Scripts/Data_preparation/build_cross_splits.R script creates balanced train/validation splits:
Process:

1. Load per-tissue CSV files from data/processed/tissues/
2. Label editing sites: "yes" (≥10%) vs "no" (<1%)
3. Create balanced datasets (equal yes/no samples)
4. Generate all tissue-pair combinations for cross-validation
5. Remove training examples from validation sets to prevent data leakage

Input:

`--data_dir`: Per-tissue CSV files 
`--train_size`: Training samples per tissue (default: 19,200)
`--valid_size`: Validation samples per tissue (default: 4,800)
`--yes_cutoff`: Editing threshold for positive class (default: 10%)
`--no_cutoff`: Non-editing threshold for negative class (default: 1%)

Output:

Cross-tissue directories → data/processed/cross_splits/{train_tissue}/{train_tissue}_{valid_tissue}/
Training files: {train_tissue}_train.csv
Validation files: {valid_tissue}_valid.csv
Summary report: cross_split_summary.csv

```
Rscript scripts/build_cross_splits.R \
    --data_dir data/processed/tissues/ \
    --output_dir data/processed/cross_splits/ \
    --train_size 19200 \
    --valid_size 4800 \
    --yes_cutoff 10 \
    --no_cutoff 1 \
    --seed 42
```


## Model Training and Evaluation
### Model Architecture
ADAREDIT employs a Graph Attention Network (GAT) architecture with the following components:

Graph Representation: RNA segments as graphs with nucleotides as nodes
Edge Types: Sequential (adjacent nucleotides) and structural (base-pairs)
Node Features: 8-dimensional vectors including base encoding, pairing status, relative position, and target flag
Architecture: 3-layer GAT with multi-head attention (4 heads per layer)
Output: Binary classification with attention weights for interpretability

Training a Model
Basic Training Command:

```
python Scripts/model/gnnadar_verb_compact.py \
    --train_file {tissue}_train.csv \
    --val_file {tissue}_valid.csv \
    --epochs 1000 \
    --mode train \
    --batch_size 128 \
    --num_workers 6 \
    --checkpoint_dir checkpoints/Liver \
    --checkpoint_interval 10
```
Training Parameters:

`--train_file`: Path to training CSV file
`--val_file`: Path to validation CSV file
`--epochs`: Number of training epochs (default: 600)
`--batch_size`: Training batch size (default: 128)
`--mode`: Operation mode ('train' or 'eval')
`--checkpoint_dir`: Directory to save model checkpoints
`--checkpoint_interval`: Epoch interval for saving checkpoints (default: 10)

Model Evaluation
Evaluate Pre-trained Model:

```
python Scripts/model/gnnadar_verb_compact.py \
    --val_file {{tissue}_valid.csv \
    --mode eval \
    --checkpoint checkpoints/{tissue}/model_checkpoint_epoch_980.pth
```

## Results
### Results: Cross-Tissue and Cross-Species

![cross_tissues_evo](Figure/cross_tissues_evo.png)
Figure 2: (A) Cross-tissue evaluation showing model performance across different tissue combinations. (B) Cross-species evaluation demonstrating generalization capability







