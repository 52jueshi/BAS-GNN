"""Graph sparsity and execution-regularity metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def row_nnz(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Count incoming retained edges per node under PyG message flow."""
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    return torch.bincount(edge_index[1], minlength=num_nodes)


def compute_regularity_metrics(
    sparse_edge_index: torch.Tensor,
    num_nodes: int,
    original_num_edges: int,
    alignments: Sequence[int] = (8, 16, 32),
    eps: float = 1e-12,
) -> tuple[dict[str, float | int], torch.Tensor]:
    """Compute sparsity, row-balance, pattern, and padding metrics."""
    counts = row_nnz(sparse_edge_index, num_nodes).to(torch.float64)
    real_nnz = int(sparse_edge_index.size(1))
    mean = float(counts.mean()) if num_nodes else 0.0
    std = float(counts.std(unbiased=False)) if num_nodes else 0.0
    maximum = float(counts.max()) if num_nodes else 0.0
    retention = real_nnz / original_num_edges if original_num_edges else 0.0
    result: dict[str, float | int] = {
        "num_edges": real_nnz,
        "edge_retention_ratio": retention,
        "edge_sparsity": 1.0 - retention,
        "mean_row_nnz": mean,
        "std_row_nnz": std,
        "row_cv": std / (mean + eps),
        "max_mean_ratio": maximum / (mean + eps),
        "effective_row_width_count": int(torch.unique(counts).numel()),
        # Backward-compatible Phase-1 alias. This is physical row-width
        # diversity, not the number of learned templates.
        "pattern_complexity": int(torch.unique(counts).numel()),
    }
    for alignment in alignments:
        if alignment <= 0:
            raise ValueError("alignment values must be positive")
        aligned = int((torch.ceil(counts / alignment) * alignment).sum())
        padding_ratio = 0.0 if aligned == 0 else 1.0 - real_nnz / aligned
        result[f"aligned_nnz_p{alignment}"] = aligned
        result[f"padding_ratio_p{alignment}"] = padding_ratio
    return result, counts.to(torch.long)
