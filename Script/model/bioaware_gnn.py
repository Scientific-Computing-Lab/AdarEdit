#!/usr/bin/env python
"""
Biology-aware GNN for A-to-I RNA editing.

Key design choices:
- Node features: base one-hot, paired flag, relative position to target (normalized),
  target flag, optional neighbor one-hots, stem/loop geometry, pair energy/type.
- Edge features: learned edge-type embedding (sequence / canonical pair / wobble pair)
  plus scalar attributes (seq vs pair flag, sequence distance, geometry on both ends,
  pair energies, distance to target indices).
- Sequence branch: small 1D CNN over the raw tokenized sequence to capture local
  trinucleotide motifs known to influence ADAR binding/editing.
- Fusion: graph-pooled embedding concatenated with sequence CNN embedding,
  passed through an MLP head for binary classification.
"""

from __future__ import annotations

import json
import math
import copy
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# --------------------------
#   Constants / lookups
# --------------------------
BASES = ["A", "G", "C", "U", "N", "PAD"]
BASE_IDX: Dict[str, int] = {b: i for i, b in enumerate(BASES)}

SEQ_EDGE = 0
CANONICAL_EDGE = 1
WOBBLE_EDGE = 2

CANONICAL = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
WOBBLE = {("G", "U"), ("U", "G")}

# Approximate nearest-neighbor derived pairing energies (kcal/mol, coarse)
PAIR_ENERGY = {
    ("A", "U"): -2.1,
    ("U", "A"): -2.1,
    ("G", "C"): -3.4,
    ("C", "G"): -3.4,
    ("G", "U"): -1.0,
    ("U", "G"): -1.0,
}


def classify_pair(b1: str, b2: str) -> int:
    pair = (b1, b2)
    if pair in CANONICAL:
        return CANONICAL_EDGE
    if pair in WOBBLE:
        return WOBBLE_EDGE
    return SEQ_EDGE


def paired_map(struct: str) -> Dict[int, int]:
    stack, pm = [], {}
    for i, c in enumerate(struct):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            j = stack.pop()
            pm[i] = j
            pm[j] = i
    return pm


def stem_loop_geometry(struct: str) -> Tuple[List[int], List[int], List[int]]:
    n = len(struct)
    pm = paired_map(struct)
    stem_len = [0] * n
    loop_len = [0] * n
    dist_junc = [0] * n

    for i in range(n):
        if i in pm:
            l = i
            while l - 1 >= 0 and (l - 1) in pm and pm.get(l - 1, -10) == pm.get(l, -10) - 1:
                l -= 1
            r = i
            while r + 1 < n and (r + 1) in pm and pm.get(r + 1, -10) == pm.get(r, -10) + 1:
                r += 1
            stem_len[i] = r - l + 1
        else:
            l = i
            while l - 1 >= 0 and struct[l - 1] == ".":
                l -= 1
            r = i
            while r + 1 < n and struct[r + 1] == ".":
                r += 1
            loop_len[i] = r - l + 1

        # distance to junction (change between paired/unpaired)
        d_left = 0
        j = i
        while j > 0 and struct[j] == struct[j - 1]:
            j -= 1
            d_left += 1
        d_right = 0
        j = i
        while j < n - 1 and struct[j] == struct[j + 1]:
            j += 1
            d_right += 1
        dist_junc[i] = min(d_left, d_right)
    return stem_len, loop_len, dist_junc


# --------------------------
#   Data parsing helpers
# --------------------------
def parse_jsonl_line(js: dict) -> Tuple[str, str, int, int]:
    user_msg = next(m["content"] for m in js["messages"] if m["role"] == "user")
    label = 1 if next(m["content"] for m in js["messages"] if m["role"] == "assistant").strip().lower() == "yes" else 0
    parts = {kv.split(":")[0]: kv.split(":")[1] for kv in user_msg.split(", ")}
    L, A, R = parts.get("L", ""), parts.get("A", ""), parts.get("R", "")
    struct = parts.get("Alu Vienna Structure", "")
    seq = (L + A + R).replace("T", "U")
    target = len(L)
    return seq, struct, target, label


@dataclass
class GraphExample:
    seq: str
    struct: str
    target: int
    label: int

    @property
    def length(self) -> int:
        return len(self.seq)


def load_graphs(jsonl_path: str, *, use_neighbors: bool, use_geometry: bool, plfold_dir: str = None) -> List[Data]:
    graphs: List[Data] = []
    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            js = json.loads(line)
            seq, struct, tgt, label = parse_jsonl_line(js)
            if len(seq) != len(struct):
                continue
            g = build_graph(GraphExample(seq, struct, tgt, label), use_neighbors, use_geometry, plfold_dir=plfold_dir, idx=idx)
            graphs.append(g)
    return graphs


def build_graph(ex: GraphExample, use_neighbors: bool, use_geometry: bool, plfold_dir: str = None, idx: int = None) -> Data:
    n = ex.length
    pm = paired_map(ex.struct)
    stem_len, loop_len, dist_junc = stem_loop_geometry(ex.struct) if use_geometry else ([0] * n, [0] * n, [0] * n)

    plfold_unpaired = None
    plfold_paired = None
    if plfold_dir is not None and idx is not None:
        npz_path = os.path.join(plfold_dir, f"seq_{idx:06d}.npz")
        if os.path.isfile(npz_path):
            try:
                data = np.load(npz_path)
                plfold_unpaired = data['unpaired']
                plfold_paired = data['paired']
            except Exception:
                plfold_unpaired = None
                plfold_paired = None

    x_rows = []
    seq_ids = []
    pair_type_ids = []

    for i, base in enumerate(ex.seq):
        base = base.upper()
        base_idx = BASE_IDX.get(base, BASE_IDX["N"])
        base_oh = [1 if k == base_idx else 0 for k in range(5)]  # exclude PAD token

        paired_flag = [1 if i in pm else 0]
        rel_pos = [(i - ex.target) / max(1, n)]
        target_flag = [1 if i == ex.target else 0]

        feats = base_oh + paired_flag + rel_pos + target_flag

        if use_neighbors:
            left = ex.seq[i - 1].upper() if i - 1 >= 0 else "N"
            right = ex.seq[i + 1].upper() if i + 1 < n else "N"
            feats += [1 if k == BASE_IDX.get(left, BASE_IDX["N"]) else 0 for k in range(5)]
            feats += [1 if k == BASE_IDX.get(right, BASE_IDX["N"]) else 0 for k in range(5)]

        if use_geometry:
            feats += [float(stem_len[i]), float(loop_len[i]), float(dist_junc[i])]

        pair_energy = PAIR_ENERGY.get((base, ex.seq[pm[i]].upper()), 0.0) if i in pm else 0.0
        feats += [pair_energy]

        if plfold_unpaired is not None and len(plfold_unpaired) == n:
            feats += [float(plfold_unpaired[i]), float(plfold_paired[i])]

        ptype = classify_pair(base, ex.seq[pm[i]].upper()) if i in pm else SEQ_EDGE
        pair_type_ids.append(ptype)

        x_rows.append(feats)
        seq_ids.append(base_idx)

    x = torch.tensor(x_rows, dtype=torch.float)
    seq_ids_t = torch.tensor(seq_ids, dtype=torch.long)
    pair_type_t = torch.tensor(pair_type_ids, dtype=torch.long)

    edge_index: List[List[int]] = []
    edge_type: List[int] = []
    edge_scalar: List[List[float]] = []

    def target_prox(idx: int) -> float:
        return abs(idx - ex.target) / max(1, n)

    def add_edge(i: int, j: int, et: int, is_seq: float, dist_norm: float):
        edge_index.append([i, j])
        edge_type.append(et)
        edge_scalar.append(
            [
                is_seq,
                1.0 - is_seq,  # is_pair flag
                dist_norm,
                float(stem_len[i]),
                float(stem_len[j]),
                float(loop_len[i]),
                float(loop_len[j]),
                float(dist_junc[i]),
                float(dist_junc[j]),
                float(PAIR_ENERGY.get((ex.seq[i].upper(), ex.seq[j].upper()), 0.0)),
                target_prox(i),
                target_prox(j),
            ]
        )

    for i in range(n - 1):
        dist = 1.0 / max(1, n)
        add_edge(i, i + 1, SEQ_EDGE, 1.0, dist)
        add_edge(i + 1, i, SEQ_EDGE, 1.0, dist)

    for i, j in pm.items():
        if i < j:
            et = classify_pair(ex.seq[i].upper(), ex.seq[j].upper())
            add_edge(i, j, et, 0.0, 0.0)
            add_edge(j, i, et, 0.0, 0.0)

    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type_t = torch.tensor(edge_type, dtype=torch.long)
    edge_scalar_t = torch.tensor(edge_scalar, dtype=torch.float)
    y = torch.tensor([ex.label], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index_t,
        edge_type=edge_type_t,
        edge_scalar=edge_scalar_t,
        y=y,
        seq_ids=seq_ids_t,
        pair_type=pair_type_t,
    )


# --------------------------
#   Model components
# --------------------------
class SeqCNN(nn.Module):
    def __init__(self, emb_dim: int = 12, channels: int = 48):
        super().__init__()
        self.emb = nn.Embedding(len(BASES), emb_dim, padding_idx=BASE_IDX["PAD"])
        self.conv3 = nn.Conv1d(emb_dim, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(emb_dim, channels, kernel_size=5, padding=2)
        self.proj = nn.Linear(channels * 2, channels)

    def forward(self, seq_batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # seq_batch: [B, L] (padded); mask: [B, L]
        emb = self.emb(seq_batch).transpose(1, 2)  # [B, C, L]
        c3 = F.relu(self.conv3(emb))
        c5 = F.relu(self.conv5(emb))
        # masked pooling
        mask = mask.unsqueeze(1)  # [B,1,L]
        c3 = c3.masked_fill(~mask.bool(), float("-inf"))
        c5 = c5.masked_fill(~mask.bool(), float("-inf"))
        p3 = torch.max(c3, dim=2).values
        p5 = torch.max(c5, dim=2).values
        return F.relu(self.proj(torch.cat([p3, p5], dim=1)))


class BioAwareGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 96,
        heads: int = 4,
        layers: int = 3,
        edge_emb_dim: int = 6,
        edge_scalar_dim: int = 12,
        dropout: float = 0.1,
        seq_branch_dim: int = 48,
        use_global_attn: bool = False,
        global_attn_heads: int = 4,
    ):
        super().__init__()
        self.use_seq_branch = seq_branch_dim and seq_branch_dim > 0
        self.use_global_attn = use_global_attn
        self.edge_emb = nn.Embedding(3, edge_emb_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_emb_dim + edge_scalar_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        gat_edge_dim = 16

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(
            GATv2Conv(in_dim, hidden, heads=heads, concat=True, edge_dim=gat_edge_dim, dropout=dropout)
        )
        self.bns.append(nn.BatchNorm1d(hidden * heads))
        for _ in range(layers - 1):
            self.convs.append(
                GATv2Conv(hidden * heads, hidden, heads=heads, concat=True, edge_dim=gat_edge_dim, dropout=dropout)
            )
            self.bns.append(nn.BatchNorm1d(hidden * heads))
        self.dropout = nn.Dropout(dropout)

        if self.use_seq_branch:
            self.seq_encoder = SeqCNN(channels=seq_branch_dim)
            seq_dim = seq_branch_dim
        else:
            seq_dim = 0

        self.global_attn = None
        if self.use_global_attn:
            self.global_attn = nn.MultiheadAttention(embed_dim=hidden * heads, num_heads=global_attn_heads, batch_first=True)

        fusion_dim = hidden * heads + seq_dim + (hidden * heads if self.use_global_attn else 0)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _build_seq_batch(self, seq_ids: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_idx = BASE_IDX["PAD"]
        seq_batch, mask = to_dense_batch(seq_ids, batch, fill_value=pad_idx)
        return seq_batch, mask.float()

    def forward(self, data: Data) -> torch.Tensor:
        ea = self.edge_emb(data.edge_type)
        ecat = torch.cat([ea, data.edge_scalar], dim=1)
        edge_attr = self.edge_mlp(ecat)

        x, ei, bs = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei, edge_attr=edge_attr)
            x = F.relu(x)
            x = bn(x)
            x = self.dropout(x)
        g_emb = global_mean_pool(x, bs)

        g_global = None
        if self.use_global_attn:
            x_dense, mask = to_dense_batch(x, bs)  # [B, L, D]
            # key_padding_mask uses False for valid, True for pad
            attn_out, _ = self.global_attn(x_dense, x_dense, x_dense, key_padding_mask=~mask)
            # masked mean
            mask_f = mask.float().unsqueeze(-1)
            g_global = (attn_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)

        if self.use_seq_branch:
            seq_batch, mask = self._build_seq_batch(data.seq_ids, bs)
            seq_emb = self.seq_encoder(seq_batch, mask)
            parts = [g_emb, seq_emb]
        else:
            parts = [g_emb]

        if g_global is not None:
            parts.append(g_global)

        fused = torch.cat(parts, dim=1)
        return self.head(fused).squeeze()


# --------------------------
#   Training / evaluation
# --------------------------
def evaluate(model: nn.Module, loader: DataLoader, threshold: float = 0.5):
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            y_prob.extend(probs.cpu().tolist())
            y_true.extend(batch.y.cpu().tolist())
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    spec = recall_score(y_true, y_pred, pos_label=0)
    prec = precision_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return acc, f1, rec, spec, prec, auc, y_prob, y_true


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 300,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    patience: int = 40,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    best = {"f1": 0.0, "state": None, "epoch": 0, "auc": float("nan")}
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y.squeeze())
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()

        acc, f1, rec, spec, prec, auc, _, _ = evaluate(model, val_loader)
        avg_loss = total / max(1, len(train_loader))
        print(f"Epoch {ep:03d}  loss={avg_loss:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

        if f1 > best["f1"] + 1e-4:
            best.update({"f1": f1, "state": copy.deepcopy(model.state_dict()), "epoch": ep, "auc": auc})
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print("Early stopping.")
            break
    return best


__all__ = [
    "BioAwareGNN",
    "SeqCNN",
    "load_graphs",
    "evaluate",
    "train_model",
    "build_graph",
]
