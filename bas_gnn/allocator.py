"""Three-width Lagrangian search, retaining the experimental tie policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


QUOTAS = (4, 8, 16)


@dataclass(frozen=True)
class BudgetAllocation:
    lagrangian_lambda: float
    selected_actions: torch.Tensor
    selected_quota_codes: torch.Tensor
    selected_costs: torch.Tensor
    selected_gains: torch.Tensor
    target_budget: int
    actual_budget: int
    unused_budget: int
    relative_budget_error: float


def allocate_budget(
    utilities: torch.Tensor,
    receiver_degree: torch.Tensor,
    target_budget: int,
    *,
    iterations: int = 80,
    maximum_lambda_expansions: int = 80,
) -> BudgetAllocation:
    """Search feasible Lagrangian allocations; do not force budget saturation."""
    if utilities.shape != (receiver_degree.numel(), 3):
        raise ValueError("utilities must have shape [num_nodes,3]")
    if receiver_degree.dtype != torch.long:
        raise TypeError("receiver_degree must be long")
    if not bool(torch.isfinite(utilities).all()):
        raise ValueError("predicted gains must be finite")
    costs = torch.stack(
        [
            torch.minimum(receiver_degree, torch.full_like(receiver_degree, quota))
            for quota in QUOTAS
        ],
        dim=1,
    )
    minimum_cost = int(costs[:, 0].sum())
    if target_budget < minimum_cost:
        raise ValueError("target is below the all-q4 minimum cost")

    zero_actions = actions_at_lambda(utilities, costs, 0.0)
    zero_cost = _selected_sum(costs, zero_actions)
    if zero_cost <= target_budget:
        return _result(
            utilities, costs, zero_actions, 0.0, target_budget
        )

    low = 0.0
    high = 1.0
    high_actions = actions_at_lambda(utilities, costs, high)
    expansions = 0
    while _selected_sum(costs, high_actions) > target_budget:
        high *= 2.0
        high_actions = actions_at_lambda(utilities, costs, high)
        expansions += 1
        if expansions >= maximum_lambda_expansions:
            raise RuntimeError("could not bracket a feasible BAS allocation")
    best_lambda = high
    best_actions = high_actions
    best_gain = _selected_gain(utilities, high_actions)
    best_cost = _selected_sum(costs, high_actions)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        actions = actions_at_lambda(utilities, costs, middle)
        cost = _selected_sum(costs, actions)
        if cost > target_budget:
            low = middle
        else:
            gain = _selected_gain(utilities, actions)
            if gain > best_gain + 1e-12 or (
                abs(gain - best_gain) <= 1e-12 and cost < best_cost
            ):
                best_lambda = middle
                best_actions = actions
                best_gain = gain
                best_cost = cost
            high = middle
    for candidate in (
        high,
        math.nextafter(high, math.inf),
        math.nextafter(low, math.inf),
    ):
        actions = actions_at_lambda(utilities, costs, candidate)
        cost = _selected_sum(costs, actions)
        if cost <= target_budget:
            gain = _selected_gain(utilities, actions)
            if gain > best_gain + 1e-12 or (
                abs(gain - best_gain) <= 1e-12 and cost < best_cost
            ):
                best_lambda = candidate
                best_actions = actions
                best_gain = gain
                best_cost = cost
    return _result(
        utilities, costs, best_actions, best_lambda, target_budget
    )


def actions_at_lambda(
    gains: torch.Tensor, costs: torch.Tensor, lagrangian_lambda: float
) -> torch.Tensor:
    """Choose max gain-lambda*cost; option order breaks ties toward lower cost."""
    if lagrangian_lambda < 0:
        raise ValueError("lambda must be non-negative")
    scores = gains - float(lagrangian_lambda) * costs.to(gains.dtype)
    return scores.argmax(dim=1)


def _result(
    gains: torch.Tensor,
    costs: torch.Tensor,
    actions: torch.Tensor,
    lagrangian_lambda: float,
    target_budget: int,
) -> BudgetAllocation:
    nodes = torch.arange(actions.numel(), device=actions.device)
    selected_costs = costs[nodes, actions]
    actual = int(selected_costs.sum())
    if actual > target_budget:
        raise RuntimeError("BAS allocator returned an over-budget solution")
    quota_codes = torch.tensor(QUOTAS, device=actions.device)[actions]
    return BudgetAllocation(
        float(lagrangian_lambda),
        actions,
        quota_codes,
        selected_costs,
        gains[nodes, actions],
        int(target_budget),
        actual,
        int(target_budget) - actual,
        (int(target_budget) - actual) / int(target_budget),
    )


def _selected_sum(values: torch.Tensor, actions: torch.Tensor) -> int:
    nodes = torch.arange(actions.numel(), device=actions.device)
    return int(values[nodes, actions].sum())


def _selected_gain(values: torch.Tensor, actions: torch.Tensor) -> float:
    nodes = torch.arange(actions.numel(), device=actions.device)
    return float(values[nodes, actions].sum())
