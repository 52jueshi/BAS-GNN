"""Shared inductive scorer for candidate graph edges."""

from __future__ import annotations

import torch
from torch import nn


class LocalEdgeScorer(nn.Module):
    """Score edges with a shared MLP; no edge owns a trainable parameter."""

    def __init__(self, in_channels: int, projection_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_channels, projection_channels)
        scorer_in = 4 * projection_channels + 2
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, in_degree: torch.Tensor
    ) -> torch.Tensor:
        """Return one scalar per edge.

        Shapes: ``x=[N,F]``, ``edge_index=[2,E]``, ``in_degree=[N]``, output ``[E]``.
        """
        if edge_index.dtype != torch.long or edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise TypeError("edge_index must have shape [2, E] and dtype torch.long")
        if in_degree.shape != (x.size(0),):
            raise ValueError("in_degree must have shape [num_nodes]")
        projected = self.projection(x)  # [N, P]
        source, target = edge_index
        source_x = projected[source]
        target_x = projected[target]
        features = torch.cat(
            [
                source_x,
                target_x,
                (source_x - target_x).abs(),
                source_x * target_x,
                torch.log1p(in_degree[source]).unsqueeze(-1),
                torch.log1p(in_degree[target]).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.scorer(features).squeeze(-1)
