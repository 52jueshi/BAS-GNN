"""Frozen Original-Softmax utility over the fixed {4,8,16} quota lattice."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from .graphs import build_static_sparse_graph
from .allocator import allocate_budget
from .graphs import from_v2, WeightedStaticGraph


@dataclass(frozen=True)
class StructuredImportanceAllocation:
    utilities: torch.Tensor
    selected_quota_codes: torch.Tensor
    selected_costs: torch.Tensor
    lagrangian_lambda: float
    target_budget: int
    actual_budget: int
    unused_budget: int


def build_utility(context: object) -> torch.Tensor:
    """Return U_i(q)=sum_{rank<q} softmax(score), q in {4,8,16}."""
    receiver = context.candidate_edge_index[1]
    utilities = []
    for quota in (4, 8, 16):
        keep = context.curve.edge_rank < int(quota)
        value = context.curve.edge_attention.new_zeros(context.data.num_nodes)
        value.index_add_(0, receiver[keep], context.curve.edge_attention[keep])
        utilities.append(value)
    result = torch.stack(utilities, dim=1)
    if bool((result[:, 1:] + 1e-7 < result[:, :-1]).any()):
        raise RuntimeError("structured importance utility is not monotone")
    return result


def allocate(context: object, target_budget: int, iterations: int = 80) -> StructuredImportanceAllocation:
    utilities = build_utility(context)
    raw = allocate_budget(utilities, context.degree, int(target_budget), iterations=int(iterations))
    if raw.actual_budget > int(target_budget):
        raise RuntimeError("structured allocation exceeded Fixed-q8 budget")
    if not set(raw.selected_quota_codes.detach().cpu().tolist()).issubset({4, 8, 16}):
        raise RuntimeError("structured allocation used a forbidden quota")
    return StructuredImportanceAllocation(
        utilities, raw.selected_quota_codes, raw.selected_costs,
        raw.lagrangian_lambda, raw.target_budget, raw.actual_budget, raw.unused_budget,
    )


def build_graph(context: object, allocation: StructuredImportanceAllocation) -> WeightedStaticGraph:
    classes = len(set(allocation.selected_quota_codes.detach().cpu().tolist()))
    return from_v2(build_static_sparse_graph(
        context.candidate_edge_index, context.curve.edge_rank, allocation.selected_costs,
        context.data.num_nodes, classes,
    ))
