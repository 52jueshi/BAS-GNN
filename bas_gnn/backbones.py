"""Two-layer backbones that consume the exported static BAS topology."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv

from .gcn import DenseGCN


BACKBONES = ("GCN", "GraphSAGE", "GAT")


class SparseMeanSAGEConv(nn.Module):
    """PyG-compatible mean GraphSAGE algebra without an E-by-F message tensor."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin_neighbor = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_root = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, receiver = edge_index
        num_nodes = x.size(0)
        degree = torch.bincount(receiver, minlength=num_nodes).to(x.dtype)
        values = degree[receiver].clamp_min(1.0).reciprocal()
        adjacency = torch.sparse_coo_tensor(
            torch.stack([receiver, source]),
            values,
            (num_nodes, num_nodes),
            device=x.device,
            dtype=x.dtype,
            check_invariants=False,
        ).coalesce()
        neighbor_mean = torch.sparse.mm(adjacency, x)
        return self.lin_neighbor(neighbor_mean) + self.lin_root(x)


class DenseGraphSAGE(nn.Module):
    """Two-layer mean-aggregation GraphSAGE with shared hidden width."""

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float
    ) -> None:
        super().__init__()
        self.conv1 = SparseMeanSAGEConv(in_channels, hidden_channels)
        self.conv2 = SparseMeanSAGEConv(hidden_channels, out_channels)
        self.dropout = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if edge_weight is not None:
            raise ValueError("GraphSAGE does not consume scorer importance as edge weight")
        hidden = F.relu(self.conv1(x, edge_index))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        logits = self.conv2(hidden, edge_index)
        if return_hidden:
            return logits, hidden
        return logits


class DenseGAT(nn.Module):
    """Two-layer GAT whose concatenated hidden width remains fixed."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError("GAT hidden_channels must be divisible by gat_heads")
        per_head = hidden_channels // heads
        self.conv1 = GATConv(
            in_channels,
            per_head,
            heads=heads,
            concat=True,
            dropout=float(dropout),
            add_self_loops=True,
        )
        self.conv2 = GATConv(
            hidden_channels,
            out_channels,
            heads=1,
            concat=False,
            dropout=float(dropout),
            add_self_loops=True,
        )
        self.dropout = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if edge_weight is not None:
            raise ValueError("GAT computes native attention on the exported topology")
        hidden = F.relu(self.conv1(x, edge_index))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        logits = self.conv2(hidden, edge_index)
        if return_hidden:
            return logits, hidden
        return logits


def build_backbone(
    model_config: dict[str, Any],
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    name = str(model_config.get("backbone", "GCN"))
    hidden = int(model_config["hidden_channels"])
    dropout = float(model_config["dropout"])
    if name == "GCN":
        return DenseGCN(in_channels, hidden, out_channels, dropout)
    if name == "GraphSAGE":
        return DenseGraphSAGE(in_channels, hidden, out_channels, dropout)
    if name == "GAT":
        return DenseGAT(
            in_channels,
            hidden,
            out_channels,
            dropout,
            heads=int(model_config.get("gat_heads", 4)),
        )
    raise ValueError(f"Unsupported backbone {name!r}; choose one of {BACKBONES}")
