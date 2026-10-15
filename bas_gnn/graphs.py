"""Static directed top-quota graphs; GCN supplies self-loops and normalization."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import time
import torch
from .regularity import compute_regularity_metrics

@dataclass(frozen=True)
class V2StaticGraph:
    edge_index: torch.Tensor
    row_nnz: torch.Tensor
    metrics: dict[str, Any]


def build_static_sparse_graph(
    candidate_edge_index: torch.Tensor,
    edge_rank: torch.Tensor,
    selected_budget: torch.Tensor,
    num_nodes: int,
    execution_class_count: int,
) -> V2StaticGraph:
    """Freeze Top-q_i edges and measure the resulting row structure."""
    if selected_budget.dtype != torch.long or selected_budget.shape != (num_nodes,):
        raise TypeError("selected_budget must be long with shape [num_nodes]")
    if edge_rank.shape != (candidate_edge_index.size(1),):
        raise ValueError("edge rank must align with candidate edges")
    receiver = candidate_edge_index[1]
    degree = torch.bincount(receiver, minlength=num_nodes)
    if bool((selected_budget < 0).any()) or bool((selected_budget > degree).any()):
        raise ValueError("selected node budgets must lie in [0, degree]")
    keep = edge_rank < selected_budget[receiver]
    sparse_edge_index = candidate_edge_index[:, keep]
    regularity, row_nnz = compute_regularity_metrics(
        sparse_edge_index,
        num_nodes,
        int(candidate_edge_index.size(1)),
        alignments=(16,),
    )
    if not torch.equal(row_nnz, selected_budget):
        raise RuntimeError("Frozen Top-q graph row counts differ from allocated costs")
    metrics = {
        **regularity,
        "mean_effective_quota": float(row_nnz.to(torch.float64).mean()),
        "execution_class_count": int(execution_class_count),
    }
    return V2StaticGraph(sparse_edge_index, row_nnz, metrics)


@dataclass(frozen=True)
class WeightedStaticGraph:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor | None
    row_nnz: torch.Tensor
    metrics: dict[str, float | int]
    preprocessing_seconds: float
    sampling_draws: int | None = None


def from_v2(graph: V2StaticGraph, preprocessing_seconds: float = 0.0) -> WeightedStaticGraph:
    return WeightedStaticGraph(
        graph.edge_index,
        None,
        graph.row_nnz,
        dict(graph.metrics),
        float(preprocessing_seconds),
    )


def build_fixed_graph(context: object, quota: int) -> WeightedStaticGraph:
    started = time.perf_counter()
    budget = torch.minimum(
        context.degree, torch.full_like(context.degree, int(quota))
    )
    graph = build_static_sparse_graph(
        context.candidate_edge_index,
        context.curve.edge_rank,
        budget,
        context.data.num_nodes,
        1,
    )
    return from_v2(graph, time.perf_counter() - started)
