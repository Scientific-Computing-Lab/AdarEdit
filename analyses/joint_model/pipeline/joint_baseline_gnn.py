#!/usr/bin/env python
"""
Joint baseline GNN for A-to-I RNA editing.

This is the *baseline* counterpart of the bio-aware joint model. It mirrors the
compact 8-feature sequence-GAT baseline (RNAEditingGNN from
gnnadar_verb_compact.py) used for the per-tissue and cross-tissue matrix models,
but with a configurable-width output head so it can drive either joint
multilabel task (five outputs with Combined or four outputs without Combined).

Differences from the per-tissue baseline RNAEditingGNN, all intentional:
  - output head width is `out_dim` (4 for multilabel) instead of 1;
  - forward returns RAW logits (no sigmoid) so training can use
    BCEWithLogitsLoss / masked BCE, identical to the bio-aware joint trainer;
  - the attention-weights side-output is dropped (not needed here).

Node features are the same 8 the baseline uses: base one-hot (5) + paired flag +
relative position to target + target flag. Edges are backbone + base-pair edges,
untyped and without edge attributes. The graph builder represents every
directed relation exactly once.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

# Re-export baseline graph construction so the trainer builds identical 8-dim
# graphs to the per-tissue / matrix baseline models.
from gnnadar_verb_compact import create_rna_graph  # noqa: F401

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JointBaselineGNN(nn.Module):
    """Compact 8-feature sequence-GAT baseline with an out_dim-wide head.

    Architecture matches RNAEditingGNN (3-layer GATConv, BatchNorm, dropout,
    global mean pool, linear head), but emits raw logits of width `out_dim`.
    """

    def __init__(self, input_dim=8, hidden_dim=32, output_dim=5, num_heads=4,
                 num_layers=3, dropout_rate=0.2):
        super().__init__()
        self.num_layers = num_layers

        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=num_heads, concat=True)
        )
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATConv(in_channels=hidden_dim * num_heads, out_channels=hidden_dim,
                        heads=num_heads, concat=True)
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
        return self.fc(x)  # raw logits [B, out_dim]
