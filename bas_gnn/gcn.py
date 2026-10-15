"""Two-layer GCN used by both dense and sparse Phase-1 models."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class DenseGCN(nn.Module):
    """Standard two-layer GCN with optional weighted edges."""

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=True)
        self.dropout = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run GCN.

        Args:
            x: Node features with shape ``[N, F]``.
            edge_index: Directed edges with shape ``[2, E]``.
            edge_weight: Optional weights with shape ``[E]``.
            return_hidden: Return post-ReLU hidden states ``[N, H]``.
        """
        hidden = self.conv1(x, edge_index, edge_weight=edge_weight)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        logits = self.conv2(hidden, edge_index, edge_weight=edge_weight)
        if return_hidden:
            return logits, hidden
        return logits
