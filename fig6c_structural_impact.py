#!/usr/bin/env python
"""Figure 6C — Structural impact across [-40, +40] (topology-aware square plot).

For each relative position in [-40, +40], compute:
  impact(pos) = pred(paired) - pred(unpaired)

Topology-aware unpaired:
- set paired-flag=0
- if a structural partner exists, remove the structural edge (both directions)
  and also set the partner paired-flag=0

Inputs:
- --val_file: validation JSONL (OpenAI-style format used in this repository)
- --checkpoint: baseline model checkpoint (.pth)

Output:
- fig6c_structural_impact_square.png
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
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

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


def run_structural_impact_square_plot(model, val_loader, device, num_limit=4000):
    full_range = range(-40, 41)
    impact = {pos: [] for pos in full_range}

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

                for pos in full_range:
                    if pos not in node_map:
                        continue
                    node_idx = node_map[pos]

                    # Paired
                    x_p = x_orig.clone()
                    x_p[node_idx, 5] = 1.0
                    graph.x = x_p
                    graph.edge_index = edge_orig
                    out_p, _ = model(graph)

                    # Unpaired (topology-aware)
                    x_u = x_orig.clone()
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

                    impact[pos].append(out_p.item() - out_u.item())

                graph.x = x_orig
                graph.edge_index = edge_orig

    xs = sorted(list(full_range))
    ys = []
    errs = []
    for x in xs:
        vals = impact[x]
        if vals:
            ys.append(float(np.mean(vals)))
            errs.append(float(np.std(vals) / np.sqrt(len(vals))))
        else:
            ys.append(0.0)
            errs.append(0.0)

    ys = np.array(ys)
    errs = np.array(errs)

    COLOR_STEM = "#2c7bb6"
    COLOR_LOOP = "#d7191c"

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(xs, ys, color="#333333", linewidth=2.5, zorder=10)

    ax.fill_between(xs, 0, ys, where=(ys >= 0), color=COLOR_STEM, alpha=0.3, interpolate=True, label="Prefers stem")
    ax.fill_between(xs, 0, ys, where=(ys <= 0), color=COLOR_LOOP, alpha=0.3, interpolate=True, label="Prefers loop")
    ax.fill_between(xs, ys - errs, ys + errs, color="black", alpha=0.1, zorder=5)

    ax.axhline(0, color="black", linestyle=":", linewidth=1.2, alpha=0.7)

    sns.despine(top=True, right=True)
    ax.set_xlim(-40, 40)
    ax.set_xlabel("Relative position", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Structural impact (paired - unpaired)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title("", fontsize=16, fontweight="bold", pad=15)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(loc="upper right", frameon=False, fontsize=12)

    plt.tight_layout()
    out = "fig6c_structural_impact_square.png"
    plt.savefig(out, dpi=600, bbox_inches="tight")
    print(f"[Fig6C] Saved: {out} (samples used: {sample_count})")


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

    run_structural_impact_square_plot(model, val_loader, device, num_limit=args.num_samples)
