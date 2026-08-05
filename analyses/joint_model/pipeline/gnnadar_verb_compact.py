#!/usr/bin/env python3
"""Compact baseline GAT with homogeneous graph construction.

The builder represents each directed backbone/base-pair relation exactly once.
Both T and U map to the same uracil/thymine one-hot channel.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool


BASE_MAP = {
    "A": [1, 0, 0, 0, 0],
    "G": [0, 1, 0, 0, 0],
    "C": [0, 0, 1, 0, 0],
    "U": [0, 0, 0, 1, 0],
    "T": [0, 0, 0, 1, 0],
    "N": [0, 0, 0, 0, 1],
}


class RNAEditingGNN(nn.Module):
    """Architecture- and checkpoint-compatible compact baseline GAT."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_heads,
        num_layers=3,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.gat_layers = nn.ModuleList(
            [
                GATConv(
                    in_channels=input_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=True,
                )
            ]
        )
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim * num_heads,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=True,
                )
            )
        self.fc = nn.Linear(hidden_dim * num_heads, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim * num_heads) for _ in range(num_layers)]
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for index in range(self.num_layers):
            x = self.gat_layers[index](x, edge_index)
            x = F.relu(x)
            x = self.batch_norm_layers[index](x)
            x = self.dropout(x)
        pooled = global_mean_pool(x, batch)
        output = torch.sigmoid(self.fc(pooled))
        _, attention_weights = self.gat_layers[0](
            data.x, data.edge_index, return_attention_weights=True
        )
        return output, attention_weights


def parse_openai_json(json_data):
    """Return sequence, dot-bracket structure, target index, and binary label."""
    try:
        user_message = next(
            message["content"]
            for message in json_data["messages"]
            if message["role"] == "user"
        )
        answer = next(
            message["content"]
            for message in json_data["messages"]
            if message["role"] == "assistant"
        )
    except Exception:
        return None, None, None, None
    fields = {}
    for field in user_message.split(", "):
        if ":" in field:
            key, value = field.split(":", 1)
            fields[key] = value
    left = fields.get("L", "")
    centre = fields.get("A", "")
    right = fields.get("R", "")
    sequence = (left + centre + right).upper()
    sequence = "".join(base if base in BASE_MAP else "N" for base in sequence)
    structure = fields.get("Alu Vienna Structure", "")
    label = 1 if answer.strip().lower() == "yes" else 0
    return sequence, structure, len(left), label


def create_rna_graph(sequence, structure, target_index, label):
    """Create a graph with one copy of every directed relation."""
    sequence = sequence.upper()
    number_nodes = len(sequence)
    if number_nodes == 0 or number_nodes != len(structure):
        return None

    stack = []
    pairs = {index: -1 for index in range(number_nodes)}
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")" and stack:
            partner = stack.pop()
            pairs[index] = partner
            pairs[partner] = index

    features = []
    for index, base in enumerate(sequence):
        features.append(
            BASE_MAP.get(base, BASE_MAP["N"])
            + [
                int(pairs[index] != -1),
                index - target_index,
                int(index == target_index),
            ]
        )

    directed_edges = set()
    for index in range(number_nodes - 1):
        directed_edges.update({(index, index + 1), (index + 1, index)})
    for index, partner in pairs.items():
        if partner != -1 and index < partner:
            directed_edges.update({(index, partner), (partner, index)})
    ordered_edges = sorted(directed_edges)
    edge_index = torch.tensor(ordered_edges, dtype=torch.long).t().contiguous()

    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.float32),
    )


def load_jsonl_to_graphs(file_path):
    graphs = []
    with Path(file_path).open() as handle:
        for line in handle:
            parsed = parse_openai_json(json.loads(line))
            sequence, structure, target_index, label = parsed
            if sequence and structure:
                graph = create_rna_graph(
                    sequence, structure, target_index, label
                )
                if graph is not None:
                    graphs.append(graph)
    return graphs


__all__ = [
    "RNAEditingGNN",
    "parse_openai_json",
    "create_rna_graph",
    "load_jsonl_to_graphs",
]
