#!/usr/bin/env python
"""Figure 6A+B — Core motif preference heatmaps (topology-aware).

This script generates TWO heatmaps over the motif window [-3, +3] (relative to the target A at 0):

Figure 6A (sequence preference):
- Mutate the base identity at each position while keeping structure/edges unchanged.

Figure 6B (structure preference, topology-aware):
- Compare paired vs unpaired at each position/base.
- In the unpaired condition, we set the paired-flag to 0 AND remove the structural edge
  to the partner node (when such partner exists).

Inputs:
- --val_file: validation JSONL (OpenAI-style format used in this repository)
- --checkpoint: baseline model checkpoint (.pth)

Outputs:
- fig6a_motif_sequence_preference_heatmap.png
- fig6b_structure_preference_heatmap.png
"""

import json
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


sns.set(style="white")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNAEditingGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers=3, dropout_rate=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=num_heads, concat=True))
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATConv(in_channels=hidden_dim * num_heads, out_channels=hidden_dim, heads=num_heads, concat=True)
            )
        self.fc = nn.Linear(hidden_dim * num_heads, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim * num_heads) for _ in range(num_layers)])

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
    num_nodes = len(sequence)
    if num_nodes != len(structure):
        return None

    base_map = {"A": [1, 0, 0, 0, 0], "G": [0, 1, 0, 0, 0], "C": [0, 0, 1, 0, 0], "T": [0, 0, 0, 1, 0], "N": [0, 0, 0, 0, 1]}
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


def generate_core_heatmaps_topology_aware(model, val_loader, device, num_limit=4000):
    core_range = range(-3, 4)
    bases = ["A", "G", "C", "T"]
    base_to_idx = {"A": 0, "G": 1, "C": 2, "T": 3}

    seq_scores = {pos: {b: [] for b in bases} for pos in core_range}
    struct_deltas = {pos: {b: [] for b in bases} for pos in core_range}

    model.eval()
    sample_count = 0

    with torch.no_grad():
        for batch in val_loader:
            if sample_count >= num_limit:
                break

            for graph in batch.to_data_list():
                if sample_count >= num_limit:
                    break
                if graph.y.item() != 1:
                    continue

                graph = graph.to(device)
                orig_out, _ = model(graph)
                if orig_out.item() < 0.7:
                    continue

                sample_count += 1

                x_orig = graph.x.clone()
                edge_orig = graph.edge_index.clone()

                rel_pos = x_orig[:, 6].cpu().numpy()
                node_map = {int(p): idx for idx, p in enumerate(rel_pos)}

                partner_map = {}
                src, dst = edge_orig.cpu().numpy()
                for u, v in zip(src, dst):
                    if abs(u - v) > 1:
                        partner_map[u] = v
                        partner_map[v] = u

                for pos in core_range:
                    if pos not in node_map:
                        continue
                    node_idx = node_map[pos]
                    orig_struct_flag = float(x_orig[node_idx, 5].item())

                    for base in bases:
                        # Sequence preference (mutate base only)
                        x_seq = x_orig.clone()
                        x_seq[node_idx, 0:5] = 0
                        x_seq[node_idx, base_to_idx[base]] = 1
                        x_seq[node_idx, 5] = orig_struct_flag
                        graph.x = x_seq
                        graph.edge_index = edge_orig
                        out_seq, _ = model(graph)
                        seq_scores[pos][base].append(out_seq.item())

                        # Structure preference (paired vs unpaired; unpaired removes partner edge)
                        x_p = x_seq.clone()
                        x_p[node_idx, 5] = 1.0
                        graph.x = x_p
                        graph.edge_index = edge_orig
                        out_p, _ = model(graph)

                        x_u = x_seq.clone()
                        x_u[node_idx, 5] = 0.0
                        edges_u = edge_orig.clone()
                        if node_idx in partner_map:
                            partner_idx = partner_map[node_idx]
                            mask = ~((edges_u[0] == node_idx) & (edges_u[1] == partner_idx))
                            mask &= ~((edges_u[0] == partner_idx) & (edges_u[1] == node_idx))
                            edges_u = edges_u[:, mask]
                            x_u[partner_idx, 5] = 0.0
                        graph.x = x_u
                        graph.edge_index = edges_u
                        out_u, _ = model(graph)

                        struct_deltas[pos][base].append(out_p.item() - out_u.item())

                graph.x = x_orig
                graph.edge_index = edge_orig

    core_sorted = sorted(list(core_range))
    mat_seq = np.zeros((4, len(core_sorted)))
    mat_struct = np.zeros((4, len(core_sorted)))
    mask = np.zeros_like(mat_seq, dtype=bool)

    for i, pos in enumerate(core_sorted):
        if pos == 0:
            mask[:, i] = True
        for j, base in enumerate(bases):
            mat_seq[j, i] = np.mean(seq_scores[pos][base]) if seq_scores[pos][base] else 0.0
            mat_struct[j, i] = np.mean(struct_deltas[pos][base]) if struct_deltas[pos][base] else 0.0

    # Column-wise normalization for the sequence heatmap (exclude target column)
    tmp = mat_seq.copy()
    idx0 = core_sorted.index(0)
    tmp[:, idx0] = np.nan
    col_means = np.nan_to_num(np.nanmean(tmp, axis=0, keepdims=True))
    mat_seq = mat_seq - col_means
    idx0_text = idx0 + 0.5

    # Fig 6A
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mat_seq, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
        xticklabels=core_sorted, yticklabels=bases, mask=mask,
        cbar_kws={"label": "Relative sequence preference"},
    )
    plt.text(idx0_text, 2, "Target\n(A)", ha="center", va="center",
             fontsize=12, fontweight="bold", color="gray", rotation=90)
    plt.title("Motif sequence preference", fontweight="bold", fontsize=14)
    plt.tight_layout()
    out_a = "fig6a_motif_sequence_preference_heatmap.png"
    plt.savefig(out_a, dpi=300)
    plt.close()

    # Fig 6B
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mat_struct, cmap="PuOr", center=0, annot=True, fmt=".2f",
        xticklabels=core_sorted, yticklabels=bases, mask=mask,
        cbar_kws={"label": "Paired - unpaired (topology-aware)"},
    )
    plt.text(idx0_text, 2, "Target\n(A)", ha="center", va="center",
             fontsize=12, fontweight="bold", color="gray", rotation=90)
    plt.title("Structure preference", fontweight="bold", fontsize=14)
    plt.tight_layout()
    out_b = "fig6b_structure_preference_heatmap.png"
    plt.savefig(out_b, dpi=300)
    plt.close()

    print(f"[Fig6A+B] Saved: {out_a}, {out_b} (samples used: {sample_count})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=4000)
    args = parser.parse_args()

    graphs = load_jsonl_to_graphs(args.val_file)
    val_loader = DataLoader(graphs, batch_size=1, shuffle=False)

    model = RNAEditingGNN(input_dim=8, hidden_dim=32, output_dim=1, num_heads=4).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)

    generate_core_heatmaps_topology_aware(model, val_loader, device, num_limit=args.num_samples)
