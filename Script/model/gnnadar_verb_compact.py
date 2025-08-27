import json
import torch
torch.manual_seed(42)  # Set a deterministic seed
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import csv
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.colors as mcolors
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import shap
import random
import seaborn as sns
import matplotlib.ticker as ticker  # For better tick formatting
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import sys
from rna_attention_analysis import run_analysis



# Global checkpoint variable
checkpoint_path = "checkpoints/model_checkpoint_epoch_980.pth"

epochsnum = 20

# **Set CPU for Training**
#torch.set_num_threads(8)
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

  
# Base mapping to include A, G, C, T, and N
base_map = {
    'A': [1, 0, 0, 0, 0],  # Adenine
    'G': [0, 1, 0, 0, 0],  # Guanine
    'C': [0, 0, 1, 0, 0],  # Cytosine
    'U': [0, 0, 0, 1, 0],
    'T': [0, 0, 0, 1, 0],  # Thymine
    'N': [0, 0, 0, 0, 1],  # Unknown or missing base
}
  
# Base colors for visualization
base_colors = {
    'A': "red",    # Adenine
    'G': "green",  # Guanine
    'C': "yellow", # Cytosine
    'T': "yellow",
    'U': "blue",   # Thymine
    'N': "gray",   # Unknown/missing bases
}


class RNAEditingGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers=3, dropout_rate=0.2):
        super(RNAEditingGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
  
        # Initial GAT layer
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=num_heads, concat=True))
  
        # Additional GAT layers
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(in_channels=hidden_dim * num_heads, out_channels=hidden_dim, heads=num_heads, concat=True))
  
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim * num_heads, output_dim)
  
        # Dropout and batch normalization
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim * num_heads) for _ in range(num_layers)])
  
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
  
        # Pass through GAT layers
        for i in range(self.num_layers):
            x = self.gat_layers[i](x, edge_index)
            x = F.relu(x)
            x = self.batch_norm_layers[i](x)
            x = self.dropout(x)
  
        # Global pooling and final prediction
        x = global_mean_pool(x, batch)
        x = self.fc(x)
  
        # Extract attention weights (from first layer)
        _, attention_weights = self.gat_layers[0](data.x, data.edge_index, return_attention_weights=True)
        return torch.sigmoid(x), attention_weights  # Return both prediction and attention

def parse_openai_json(json_data):
    """  
    Parse JSON data and extract RNA sequence, structure, target index, and label.  
  
    Parameters:  
        json_data (dict): A single JSON object containing RNA sequence information.  
  
    Returns:  
        tuple: (full_seq, structure, target_index, label) or (None, None, None, None) on error.  
    """
    try:
        user_message = next(msg["content"] for msg in json_data["messages"] if msg["role"] == "user")
        label = 1 if next(msg["content"] for msg in json_data["messages"] if msg["role"] == "assistant") == "yes" else 0
    except Exception as e:
        return None, None, None, None
  
    parts = {kv.split(":")[0]: kv.split(":")[1] for kv in user_message.split(", ")}
    left_seq = parts.get("L", "")
    central_a = parts.get("A", "")
    right_seq = parts.get("R", "")
    structure = parts.get("Alu Vienna Structure", "")
  
    # Filter the sequence to include only valid bases (A, G, C, T, N)
    valid_bases = {'A', 'G', 'C', 'T', 'N'}
    full_seq = ''.join([base if base in valid_bases else 'N' for base in left_seq + central_a + right_seq])
    target_index = len(left_seq)
  
    return full_seq, structure, target_index, label

# **Convert Data to Graph Format**
def create_rna_graph(sequence, structure, target_index, label):
    """  
    Convert RNA sequence and structure into a graph format.  
  
    Parameters:  
        sequence (str): RNA sequence containing A, G, C, T, N.  
        structure (str): RNA secondary structure in dot-bracket notation.  
        target_index (int): Index of the editing site.  
        label (int): Binary label indicating whether the site is edited (1) or not edited (0).  
  
    Returns:  
        Data: Graph representation of the RNA sequence.  
    """
    num_nodes = len(sequence)
    if num_nodes != len(structure):
        return None
  
    # Updated base mapping
    base_map = {
        'A': [1, 0, 0, 0, 0],  # Adenine
        'G': [0, 1, 0, 0, 0],  # Guanine
        'C': [0, 0, 1, 0, 0],  # Cytosine
        'U': [0, 0, 0, 1, 0],
        'T': [0, 0, 0, 1, 0],  # Thymine
        'N': [0, 0, 0, 0, 1]   # Unknown or missing base
    }
  
    x = []
    stack = []
    paired_map = {i: -1 for i in range(num_nodes)}
  
    # Parse structure to find paired bases
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")" and stack:
            j = stack.pop()
            paired_map[i] = j
            paired_map[j] = i
  
    # Create node features
    for i, base in enumerate(sequence):
        base_encoding = base_map.get(base, [0, 0, 0, 0, 1])  # Default to N if base is unknown
        paired_status = [1 if paired_map[i] != -1 else 0]
        position_encoding = [i - target_index]
        target_flag = [1 if i == target_index else 0]
        x.append(base_encoding + paired_status + position_encoding + target_flag)
  
    x = torch.tensor(x, dtype=torch.float)
  
    # Create edges
    edge_index = []
    for i in range(num_nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    for i, j in paired_map.items():
        if j != -1:
            edge_index.append([i, j])
            edge_index.append([j, i])
  
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    y = torch.tensor([label], dtype=torch.float)
  
    return Data(x=x, edge_index=edge_index, y=y)

def load_jsonl_to_graphs(file_path):
    """  
    Load RNA sequence and structure data from a JSONL file and convert to graph format.  
  
    Parameters:  
        file_path (str): Path to the JSONL file.  
  
    Returns:  
        list: List of graphs (torch_geometric.data.Data objects).  
    """
    graphs = []
    with open(file_path, 'r') as f:
        for line in f:
            json_data = json.loads(line)
            sequence, structure, target_index, label = parse_openai_json(json_data)
            if sequence and structure:
                graph = create_rna_graph(sequence, structure, target_index, label)
                if graph:
                    graphs.append(graph)
    return graphs

def load_csv_to_graphs(csv_path):
    graphs = []
    df = pd.read_csv(csv_path)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        left_seq = str(row['L']).strip().upper() if pd.notna(row['L']) else ""
        right_seq = str(row['R']).strip().upper() if pd.notna(row['R']) else ""
        structure = str(row['structure']).strip() if pd.notna(row['structure']) else ""
        label_yes_no = row['yes_no']
        label = 1 if label_yes_no == "yes" else 0


        sequence = left_seq + 'A' + right_seq  
        target_index = len(left_seq)

        graph = create_rna_graph(sequence, structure, target_index, label)
        if graph:
            graphs.append(graph)

    return graphs

def log_to_csv(filename, data, mode='a'):
    """  
    Log data to a CSV file.  
  
    Parameters:  
        filename (str): Path to the CSV file.  
        data (list): List of data to write as a row in the CSV file.  
        mode (str): File mode ('a' for append or 'w' for write).  
    """
    with open(filename, mode, newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    print("Saving training_logs.csv to:", os.getcwd())


def train(model, train_loader, val_loader, optimizer, loss_fn, epochs, start_epoch=0, checkpoint_interval=10, checkpoint_dir="checkpoints"):
    """  
    Train the RNAEditingGNN model.  
      
    Parameters:  
        model (torch.nn.Module): The RNAEditingGNN model.  
        train_loader (torch_geometric.data.DataLoader): DataLoader for the training dataset.  
        val_loader (torch_geometric.data.DataLoader): DataLoader for the validation dataset.  
        optimizer (torch.optim.Optimizer): Optimizer for training.  
        loss_fn (torch.nn.Module): Loss function for training.  
        epochs (int): Total number of epochs to train the model.  
        start_epoch (int): Epoch to start training from (useful when resuming from a checkpoint).  
        checkpoint_interval (int): Interval at which checkpoints are saved.  
        checkpoint_dir (str): Directory to save model checkpoints.  
    """
    model.train()  # Set model to training mode
    log_to_csv("training_logs.csv", ["Epoch", "Avg Loss"], mode='w')  # Create CSV header
  
    for epoch in range(start_epoch, epochs):
        print(epoch)
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)  # Move batch data to device
            optimizer.zero_grad()
  
            # Forward pass through the model
            output, _ = model(batch)  # Predict and extract attention weights
            output = output.squeeze()
  
            # Calculate loss
            loss = loss_fn(output, batch.y)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
  
            total_loss += loss.item()
  
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        log_to_csv("training_logs.csv", [epoch + 1, avg_loss])  # Log training metrics
  
        # Save checkpoint at every checkpoint interval
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f"{checkpoint_dir}/model_checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                'loss': avg_loss,
                'learning_rate': optimizer.param_groups[0]['lr']  # Save learning rate explicitly
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
  
        # Perform validation
        evaluate(model, val_loader, epoch + 1)


def evaluate(model, val_loader, epoch):
    """  
    Evaluate the RNAEditingGNN model on the validation dataset.  
  
    Parameters:  
        model (torch.nn.Module): The RNAEditingGNN model.  
        val_loader (torch_geometric.data.DataLoader): DataLoader for the validation dataset.  
        epoch (int): Current epoch during training.  
  
    Returns:  
        None  
    """
    model.eval()  # Set model to evaluation mode
    y_true, y_pred = [], []  # Ground truth labels and predictions
  
    with torch.no_grad():  # Disable gradient computation
        for batch in val_loader:
            batch = batch.to(device)  # Move batch data to device
  
            # Forward pass through the model
            output, _ = model(batch)  # Predict and extract attention weights
            preds = (output >= 0.5).float()  # Convert probabilities to binary predictions
  
            # Collect ground truth and predictions
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy().flatten())  # Flatten predictions to 1D array
  
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)  # Sensitivity for negative class
  
    # Log validation metrics
    log_to_csv("validation_logs.csv", [epoch, acc, f1, recall, specificity, precision], mode='a')
    print(f"Epoch {epoch} - Accuracy: {acc:.4f}, F1: {f1:.4f}, Sensitivity: {recall:.4f}, Specificity: {specificity:.4f}, Precision: {precision:.4f}")


def generate_attention_csv(val_graphs, model, save_path="attention_data.csv"):
    """  
    Generate a CSV dataset with attention values for validation graphs.  
  
    Parameters:  
        val_graphs (list): List of validation graphs.  
        model (torch.nn.Module): Trained GAT model.  
        save_path (str): Path to save the CSV file.  
  
    Returns:  
        None  
    """
    # Define the relative positions range (-600 to 600)
    relative_positions = list(range(-650, 650))
      
    # Initialize a list to store rows for the DataFrame
    data = []
  
    # Set model to evaluation mode
    model.eval()
  
    # Iterate over each validation graph
    with torch.no_grad():
        for graph_idx, graph in enumerate(val_graphs):
            # Move graph to device
            graph = graph.to(device)
              
            # Forward pass to get predictions and attention weights
            output, attention_weights = model(graph)
            prediction = (output >= 0.5).float().item()  # Binary prediction
            ground_truth = graph.y.item()  # Actual label
              
            # Initialize attention values for positions (-600 to 600)
            attention_map = {pos: None for pos in relative_positions}
              
            # Extract node features and attention weights
            node_features = graph.x.cpu().numpy()
            attn_values = attention_weights[1].cpu().numpy().mean(axis=1)  # Aggregate across heads
            edge_index = graph.edge_index.cpu().numpy()
  
            # Map attention values to relative positions
            for i, (src, dst) in enumerate(edge_index.T):
                src_rel_pos = int(node_features[src, 6])  # Relative position for source node
                if src_rel_pos in attention_map:
                    if attention_map[src_rel_pos] is None:
                        attention_map[src_rel_pos] = attn_values[i]
                    else:
                        attention_map[src_rel_pos] = max(attention_map[src_rel_pos], attn_values[i])  # Take max attention
              
            # Prepare row for the DataFrame
            row = [graph_idx] + [attention_map[pos] for pos in relative_positions] + [prediction, ground_truth]
            data.append(row)
  
    # Create a DataFrame
    column_names = ["graph_index"] + [f"pos_{pos}" for pos in relative_positions] + ["model_prediction", "ground_truth"]
    df = pd.DataFrame(data, columns=column_names)
  
    # Handle missing values (None -> NaN -> 0)
    df.fillna(0, inplace=True)
  
    # Save the DataFrame to CSV
    df.to_csv(save_path, index=False)
    print(f"Attention data saved to: {save_path}")


def calculate_performance_score(row):
    """Calculate the combined performance score for each epoch."""
    metrics = [row['Accuracy'], row['F1'], row['Sensitivity'], row['Specificity'], row['Precision']]
    return sum(metrics)  # Alternatively, use mean(metrics) if you prefer averaging.
  
def top_5_epochs(file_path, checkpoint_dir="checkpoints", save_path="top_epochs.csv"):
    """  
    Find the top 10 epochs based on performance score from the validation logs,  
    but only include epochs for which a checkpoint exists.  
  
    Parameters:  
        file_path (str): Path to the validation logs CSV file.  
        checkpoint_dir (str): Directory where checkpoint files are stored.  
        save_path (str): Path to save the filtered top epochs CSV file.  
  
    Returns:  
        DataFrame: Top epochs based on validation performance with existing checkpoints.  
    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, header=None, names=['Epoch', 'Accuracy', 'F1', 'Sensitivity', 'Specificity', 'Precision'])
      
    # Calculate the performance score for each epoch
    df['Performance_Score'] = df.apply(calculate_performance_score, axis=1)
      
    # Filter epochs that have corresponding checkpoint files
    def checkpoint_exists(epoch):
        checkpoint_file = os.path.join(checkpoint_dir, f"model_checkpoint_epoch_{int(epoch)}.pth")
        return os.path.isfile(checkpoint_file)
  
    df = df[df['Epoch'].apply(checkpoint_exists)]
  
    # Sort the DataFrame by the Performance_Score in descending order and get the top epochs
    top_epochs = df.sort_values(by='Performance_Score', ascending=False).head(10)
      
    # Save the top epochs to a CSV file
    top_epochs.to_csv(save_path, index=False)
    print(f"Top epochs saved to: {save_path}")
      
    return top_epochs[['Epoch', 'Accuracy', 'F1', 'Sensitivity', 'Specificity', 'Precision', 'Performance_Score']]
    
#######################        MAIN        #######################
  
if __name__ == "__main__":

    # Define command-line arguments
    parser = argparse.ArgumentParser(description="RNAEditingGNN Training and Evaluation Script")
    parser.add_argument('--train_file', type=str, help="Path to the training data JSONL file. Required in 'train' mode.")
    parser.add_argument('--val_file', type=str, required=True, help="Path to the validation data JSONL file.")
    parser.add_argument('--epochs', type=int, default=600, help="Number of epochs to train the model.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a checkpoint file.")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help="Mode of operation: 'train' for training, 'eval' for evaluation only.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training and evaluation.")
    parser.add_argument('--num_workers', type=int, default=6, help="Number of workers for DataLoader.")
    parser.add_argument('--pin_memory', action='store_true', help="Use pinned memory in DataLoader.")
    parser.add_argument('--checkpoint_interval', type=int, default=10, help="Epoch interval for saving checkpoints.")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Directory to save checkpoints.")
  
    # Parse the arguments
    args = parser.parse_args()
  
    # Validate arguments based on mode
    if args.mode == 'train' and not args.train_file:
        parser.error("--train_file is required in 'train' mode.")
  
    # Print confirmation for debugging
    print(f"Training file: {args.train_file}")
    print(f"Validation file: {args.val_file}")
    print(f"Mode: {args.mode}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
  
    if args.mode == 'train':
        train_graphs = load_csv_to_graphs(args.train_file)  # שינוי כאן: שימוש בפונקציה חדשה
        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=args.pin_memory)

    val_graphs = load_csv_to_graphs(args.val_file)  # שינוי כאן: שימוש בפונקציה חדשה
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    if args.train_file:
        os.chdir(os.path.dirname(args.train_file))
    else:
        os.chdir(os.path.dirname(args.val_file))

    print("Working directory changed to:", os.getcwd())
  
    # **Initialize Model, Optimizer, Loss**
    input_dim = 8  # 5 (base encoding) + 1 (paired status) + 1 (relative position) + 1 (target flag)
    model = RNAEditingGNN(input_dim=input_dim, hidden_dim=32, output_dim=1, num_heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
  
    # Checkpoint logic
    start_epoch = 0  # Default starting epoch
    if args.checkpoint:  # If a checkpoint is specified
        if os.path.isfile(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}...")
            checkpoint = torch.load(args.checkpoint, map_location=device)
  
            # Restore model state
            model.load_state_dict(checkpoint['model_state_dict'])
  
            # Restore optimizer state if in training mode
            if args.mode == 'train':
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
  
            # Restore start epoch
            start_epoch = checkpoint['epoch']
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
        else:
            print(f"Checkpoint file {args.checkpoint} not found. Starting fresh.")
  
    # **Mode-Specific Execution**
    if args.mode == 'train':
        # Create checkpoint directory if it doesn't exist
        os.makedirs(args.checkpoint_dir, exist_ok=True)
  
        # Train the model
        train(model, train_loader, val_loader, optimizer, loss_fn,
              epochs=args.epochs, start_epoch=start_epoch,
              checkpoint_interval=args.checkpoint_interval,
              checkpoint_dir=args.checkpoint_dir)
    elif args.mode == 'eval':
        # Evaluate the model
        print("Evaluating model...")
        evaluate(model, val_loader, epoch=start_epoch)
  
        # Optionally, generate attention data
        generate_attention_csv(val_graphs, model, save_path="attention_data.csv")
        analysis_output_dir = "attention_graphs"
        os.makedirs(analysis_output_dir, exist_ok=True)

        run_analysis(
            csv_file=args.val_file,
            attention_file="attention_data.csv",
            output_dir=analysis_output_dir,
            window_size=50
            )
        print("Evaluation complete.")
  
    # **Display and Save Top 10 Epochs Based on Validation Metrics**
    if args.mode == 'train':
        try:
            file_path = "validation_logs.csv"  # Update this path if needed
            save_path = "top_epochs.csv"       # File to save the top epochs
            checkpoint_dir = args.checkpoint_dir  # Directory containing checkpoints
            top_epochs = top_5_epochs(file_path, checkpoint_dir=checkpoint_dir, save_path=save_path)
            print("\nTop 10 Epochs Based on Validation Performance (With Checkpoints):")
            print(top_epochs)
        except FileNotFoundError:
            print("\nError: Validation logs file not found. Ensure the file exists at the specified path.")
        
"""  
Example Usage:  
  
1. Train Model from Scratch:  
--------------------------------  
Run the script to train the model from scratch using the training and validation datasets.  
The model will save checkpoints at specified intervals.  
  
Command:  
    python script.py --train_file "Liver_train.csv" --val_file "Liver_valid.csv" --epochs 100 --mode train  
  
2. Resume Training from a Checkpoint:  
--------------------------------------  
Resume training the model from a saved checkpoint. The script will load the model's weights,  
optimizer state, scheduler state (if available), and resume training from the last epoch saved in the checkpoint.  
  
Command:  
    python script.py --train_file "Liver_train.csv" --val_file "Liver_valid.csv" --epochs 100 --checkpoint "checkpoints/model_checkpoint_epoch_50.pth" --mode train  
  
3. Evaluate Model from a Checkpoint (No Training):  
--------------------------------------------------  
Load a checkpoint and evaluate the model on the validation dataset without further training.  
The script will also generate attention data and save it to a CSV file.  
  
Command:  
    python script.py --val_file "Liver_valid.csv" --checkpoint "checkpoints/model_checkpoint_epoch_50.pth" --mode eval  
"""
