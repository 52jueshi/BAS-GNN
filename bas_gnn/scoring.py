"""Frozen scorer inference and stable receiver-wise ranking."""
from __future__ import annotations
import torch
from .scorer import LocalEdgeScorer

def rank_edges_by_receiver(receiver: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Return deterministic zero-based ranks within each receiver segment."""
    if receiver.dim() != 1 or scores.shape != receiver.shape:
        raise ValueError("receiver and scores must be one-dimensional and equally sized")
    num_edges = receiver.numel()
    if num_edges == 0:
        return receiver.clone()
    score_order = torch.argsort(scores, descending=True, stable=True)
    receiver_order = torch.argsort(receiver[score_order], stable=True)
    grouped_order = score_order[receiver_order]
    grouped_receiver = receiver[grouped_order]
    positions = torch.arange(num_edges, device=receiver.device)
    is_start = torch.ones(num_edges, dtype=torch.bool, device=receiver.device)
    if num_edges > 1:
        is_start[1:] = grouped_receiver[1:] != grouped_receiver[:-1]
    starts = torch.where(is_start, positions, torch.zeros_like(positions))
    group_starts = torch.cummax(starts, dim=0).values
    grouped_rank = positions - group_starts
    rank = torch.empty_like(grouped_rank)
    rank[grouped_order] = grouped_rank
    return rank



@torch.no_grad()
def local_edge_scores(
    scorer: LocalEdgeScorer,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    receiver_degree: torch.Tensor,
    chunk_size: int = 100_000,
) -> torch.Tensor:
    """Evaluate the unchanged shared LocalEdgeScorer in bounded-memory chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    scorer.eval()
    projected = scorer.projection(x)
    source, receiver = edge_index
    degree_float = receiver_degree.to(dtype=x.dtype)
    parts: list[torch.Tensor] = []
    for start in range(0, edge_index.size(1), chunk_size):
        stop = min(start + chunk_size, edge_index.size(1))
        source_x = projected[source[start:stop]]
        receiver_x = projected[receiver[start:stop]]
        features = torch.cat(
            [
                source_x,
                receiver_x,
                (source_x - receiver_x).abs(),
                source_x * receiver_x,
                torch.log1p(degree_float[source[start:stop]]).unsqueeze(-1),
                torch.log1p(degree_float[receiver[start:stop]]).unsqueeze(-1),
            ],
            dim=-1,
        )
        parts.append(scorer.scorer(features).squeeze(-1))
    return torch.cat(parts) if parts else x.new_empty((0,))
