#!/usr/bin/env python
"""Figure 6D — Partner interaction topology (edge removal).

This script measures how the baseline GAT model responds to paired base changes
between a position near the target site (positions -1, 0, +1) and its opposing
(structural partner) nucleotide.

Topology-aware rule:
- If the mutated pair is invalid (not Watson–Crick or wobble), we:
  (1) set the paired-flag feature to 0, AND
  (2) remove the structural edge between the two nodes (both directions).

Inputs:
- --val_file: validation JSONL (OpenAI-style format used in this repository)
- --checkpoint: baseline model checkpoint (.pth)

Output:
- fig6d_partner_interaction_topology.png (default)
"""

import json
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# -----------------------------
# Plot styling
# -----------------------------
sns.set(style="white")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["svg.fonttype"] = "none"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1) Model + data (baseline boilerplate)
# ==========================================
class RNAEditingGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers=3, dropout_rate=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=num_heads, concat=True)
        )
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATConv(in_channels=hidden_dim * num_heads, out_channels=hidden_dim, heads=num_heads, concat=True)
            )
        self.fc = nn.Linear(hidden_dim * num_heads, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim * num_heads) for _ in range(num_layers)]
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.gat_layers[i](x, edge_index)
            x = F.relu(x)
            x = self.batch_norm_layers[i](x)
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return torch.sigmoid(x), None


def parse_openai_json(json_data):
    """Parse OpenAI-style JSON into (sequence, structure, target_index, label)."""
    try:
        user_message = next(msg["content"] for msg in json_data["messages"] if msg["role"] == "user")
        label = 1 if next(msg["content"] for msg in json_data["messages"] if msg["role"] == "assistant") == "yes" else 0
    except Exception:
        return None, None, None, None

    parts = {kv.split(":")[0]: kv.split(":")[1] for kv in user_message.split(", ")}
    left_seq = parts.get("L", "")
    central_a = parts.get("A", "")
    right_seq = parts.get("R", "")
    structure = parts.get("Alu Vienna Structure", "")

    full_seq = "".join([b if b in {"A", "G", "C", "T", "N"} else "N" for b in left_seq + central_a + right_seq])
    target_index = len(left_seq)
    return full_seq, structure, target_index, label


def create_rna_graph(sequence, structure, target_index, label):
    """Create a PyG graph with baseline node features."""
    num_nodes = len(sequence)
    if num_nodes != len(structure):
        return None

    base_map = {
        "A": [1, 0, 0, 0, 0],
        "G": [0, 1, 0, 0, 0],
        "C": [0, 0, 1, 0, 0],
        "T": [0, 0, 0, 1, 0],
        "N": [0, 0, 0, 0, 1],
    }

    paired_map = {i: -1 for i in range(num_nodes)}
    stack = []
    for i, ch in enumerate(structure):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            j = stack.pop()
            paired_map[i] = j
            paired_map[j] = i

    x = []
    for i, base in enumerate(sequence):
        base_encoding = base_map.get(base, [0, 0, 0, 0, 1])
        paired_status = [1 if paired_map[i] != -1 else 0]
        position_encoding = [i - target_index]
        target_flag = [1 if i == target_index else 0]
        x.append(base_encoding + paired_status + position_encoding + target_flag)

    x = torch.tensor(x, dtype=torch.float)

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
    graphs = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                json_data = json.loads(line)
                sequence, structure, target_index, label = parse_openai_json(json_data)
                if sequence and structure:
                    graph = create_rna_graph(sequence, structure, target_index, label)
                    if graph:
                        graphs.append(graph)
            except Exception:
                continue
    return graphs


# ==========================================
# 2) Analysis (topology-aware partner interaction)
# ==========================================
def analyze_partner_interaction_topology(
    model,
    val_loader,
    device,
    positions=(-1, 0, 1),
    save_path="fig6d_partner_interaction_topology.png",
):
    print(f"[Fig6D] Positions: {list(positions)} (invalid pairs remove the structural edge)")
    bases = ["A", "G", "C", "T"]
    base_map = {"A": 0, "G": 1, "C": 2, "T": 3}

    valid_pairs = {
        ("A", "T"), ("T", "A"),
        ("G", "C"), ("C", "G"),
        ("G", "T"), ("T", "G"),
    }

    matrices = {pos: {b1: {b2: [] for b2 in bases} for b1 in bases} for pos in positions}

    model.eval()
    sample_count = 0

    with torch.no_grad():
        for batch in val_loader:
            for graph in batch.to_data_list():
                if graph.y.item() != 1:
                    continue

                graph = graph.to(device)
                orig_out, _ = model(graph)
                if orig_out.item() < 0.7:
                    continue

                sample_count += 1
                if sample_count % 500 == 0:
                    print(f"[Fig6D] processed {sample_count} samples...")

                x_original = graph.x.clone()
                edge_index_original = graph.edge_index.clone()

                rel_positions = x_original[:, 6].cpu().numpy()
                node_map_pos_to_idx = {int(p): n for n, p in enumerate(rel_positions)}

                partner_map = {}
                src, dst = edge_index_original.cpu().numpy()
                for u, v in zip(src, dst):
                    if abs(u - v) > 1:
                        partner_map[u] = v
                        partner_map[v] = u

                for pos in positions:
                    if pos not in node_map_pos_to_idx:
                        continue
                    seq_node_idx = node_map_pos_to_idx[pos]
                    if seq_node_idx not in partner_map:
                        continue
                    opp_node_idx = partner_map[seq_node_idx]

                    for base_seq in bases:
                        if pos == 0 and base_seq != "A":
                            continue
                        for base_opp in bases:
                            x_mut = x_original.clone()
                            current_edge_index = edge_index_original.clone()

                            x_mut[seq_node_idx, 0:5] = 0
                            x_mut[seq_node_idx, base_map[base_seq]] = 1
                            x_mut[opp_node_idx, 0:5] = 0
                            x_mut[opp_node_idx, base_map[base_opp]] = 1

                            is_valid = (base_seq, base_opp) in valid_pairs
                            if is_valid:
                                x_mut[seq_node_idx, 5] = 1.0
                                x_mut[opp_node_idx, 5] = 1.0
                            else:
                                x_mut[seq_node_idx, 5] = 0.0
                                x_mut[opp_node_idx, 5] = 0.0

                                mask = ~(
                                    (current_edge_index[0] == seq_node_idx) & (current_edge_index[1] == opp_node_idx)
                                )
                                mask &= ~(
                                    (current_edge_index[0] == opp_node_idx) & (current_edge_index[1] == seq_node_idx)
                                )
                                current_edge_index = current_edge_index[:, mask]

                            graph.x = x_mut
                            graph.edge_index = current_edge_index
                            pred, _ = model(graph)
                            matrices[pos][base_seq][base_opp].append(pred.item())

                graph.x = x_original
                graph.edge_index = edge_index_original

    print(f"[Fig6D] Done. Samples used: {sample_count}")

    fig, axes = plt.subplots(1, len(positions), figsize=(6 * len(positions), 6))
    if len(positions) == 1:
        axes = [axes]

    all_vals = []
    for pos in positions:
        for b1 in bases:
            if pos == 0 and b1 != "A":
                continue
            for b2 in bases:
                vals = matrices[pos][b1][b2]
                if vals:
                    all_vals.append(np.mean(vals))
    vmin = min(all_vals) if all_vals else 0
    vmax = max(all_vals) if all_vals else 1

    for ax, pos in zip(axes, positions):
        mat = np.zeros((4, 4))
        for i, b1 in enumerate(bases):
            for j, b2 in enumerate(bases):
                vals = matrices[pos][b1][b2]
                mat[i, j] = np.mean(vals) if vals else 0.0

        mask = np.zeros_like(mat, dtype=bool)
        if pos == 0:
            mask[1:, :] = True

        sns.heatmap(
            mat,
            ax=ax,
            cmap="RdBu_r",
            center=(vmin + vmax) / 2,
            annot=True,
            fmt=".2f",
            vmin=vmin,
            vmax=vmax,
            xticklabels=bases,
            yticklabels=bases,
            mask=mask,
            cbar_kws={"label": "Prediction score"},
        )
        ax.set_title(f"Position {pos}", fontweight="bold", fontsize=14)
        ax.set_xlabel("Partner base", fontsize=12, fontweight="bold")
        ax.set_ylabel("Self base", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[Fig6D] Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", type=str, required=True, help="Validation JSONL file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint (.pth)")
    parser.add_argument("--out", type=str, default="fig6d_partner_interaction_topology.png", help="Output image path")
    args = parser.parse_args()

    print("[Fig6D] Loading data & model...")
    graphs = load_jsonl_to_graphs(args.val_file)
    val_loader = DataLoader(graphs, batch_size=1, shuffle=False)

    model = RNAEditingGNN(input_dim=8, hidden_dim=32, output_dim=1, num_heads=4).to(device)

    if os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    else:
        print(f"[Fig6D] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    analyze_partner_interaction_topology(model, val_loader, device, positions=(-1, 0, 1), save_path=args.out)
    print("[Fig6D] Finished.")
