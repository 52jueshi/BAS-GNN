"""Protocol-matched Dense Teacher and unchanged LocalEdgeScorer training."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data

from .datasets import DatasetInfo
from .gcn import DenseGCN
from .scorer import LocalEdgeScorer
from .scoring import rank_edges_by_receiver
from .runtime import set_seed


def train_dense_teacher(
    data: Data,
    info: DatasetInfo,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Train a dense GCN using validation-only checkpoint selection."""
    set_seed(seed)
    model = DenseGCN(
        info.num_features,
        int(model_config["hidden_channels"]),
        info.num_classes,
        float(model_config["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    best_state, metrics, log = _train_model(
        model,
        data,
        data.edge_index,
        optimizer,
        training_config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(output_dir / "training_log.csv", index=False)
    checkpoint = {
        "phase": "final-validation-f1-pretraining",
        "mode": "dense",
        "dataset": info.name,
        "seed": seed,
        "model_state_dict": _cpu_state(best_state),
        "model_config": dict(model_config),
        **metrics,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return checkpoint


class _ScorerTrainingModel(nn.Module):
    """Phase-1 fixed-q scorer training path without any Router or templates."""

    def __init__(
        self,
        info: DatasetInfo,
        model_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.student = DenseGCN(
            info.num_features,
            int(model_config["hidden_channels"]),
            info.num_classes,
            float(model_config["dropout"]),
        )
        self.edge_scorer = LocalEdgeScorer(
            info.num_features,
            int(model_config["edge_projection_channels"]),
            int(model_config["edge_hidden_channels"]),
        )

    def forward(
        self,
        data: Data,
        fixed_quota: int,
        edge_temperature: float,
    ) -> torch.Tensor:
        edge_index = data.edge_index
        receiver = edge_index[1]
        degree = torch.bincount(receiver, minlength=data.num_nodes)
        scores = self.edge_scorer(data.x, edge_index, degree.to(data.x.dtype))
        edge_rank = rank_edges_by_receiver(receiver, scores.detach())
        budget = torch.minimum(
            degree, torch.full_like(degree, int(fixed_quota))
        )
        hard = edge_rank < budget[receiver]
        degree_float = degree.to(scores.dtype)
        budget_fraction = (
            budget.to(scores.dtype) / degree_float.clamp_min(1.0)
        ).clamp(max=1.0)
        score_sum = scores.new_zeros(data.num_nodes)
        score_sum.index_add_(0, receiver, scores)
        score_mean = score_sum / degree_float.clamp_min(1.0)
        surrogate = budget_fraction[receiver] * torch.sigmoid(
            (scores - score_mean[receiver]) / float(edge_temperature)
        )
        weights = hard.to(scores.dtype) + surrogate - surrogate.detach()
        return self.student(data.x, edge_index, edge_weight=weights)

    @torch.no_grad()
    def hard_logits(self, data: Data, fixed_quota: int) -> torch.Tensor:
        self.eval()
        receiver = data.edge_index[1]
        degree = torch.bincount(receiver, minlength=data.num_nodes)
        scores = self.edge_scorer(
            data.x, data.edge_index, degree.to(data.x.dtype)
        )
        rank = rank_edges_by_receiver(receiver, scores)
        budget = torch.minimum(
            degree, torch.full_like(degree, int(fixed_quota))
        )
        return self.student(data.x, data.edge_index[:, rank < budget[receiver]])


def train_local_edge_scorer(
    data: Data,
    info: DatasetInfo,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    device: torch.device,
    seed: int,
    output_dir: Path,
    fixed_quota: int = 4,
    edge_temperature: float = 1.0,
) -> dict[str, Any]:
    """Reproduce Phase-1 scorer supervision at fixed q=4."""
    set_seed(seed)
    model = _ScorerTrainingModel(info, model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    best_validation = -1.0
    best_epoch = 0
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None
    log: list[dict[str, float | int]] = []
    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data, fixed_quota, edge_temperature)
        loss = F.cross_entropy(
            logits[data.train_mask], data.y[data.train_mask]
        )
        loss.backward()
        scorer_grad_norm = _gradient_norm(model.edge_scorer)
        optimizer.step()
        validation_logits = model.hard_logits(data, fixed_quota)
        validation_accuracy = _accuracy(
            validation_logits, data.y, data.val_mask
        )
        log.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.detach()),
                "validation_accuracy": validation_accuracy,
                "edge_scorer_gradient_norm": scorer_grad_norm,
            }
        )
        if validation_accuracy > best_validation + float(
            training_config.get("min_delta", 0.0)
        ):
            best_validation = validation_accuracy
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= int(training_config["patience"]):
            break
    if best_state is None:
        raise RuntimeError("LocalEdgeScorer training produced no checkpoint")
    model.load_state_dict(best_state)
    final_logits = model.hard_logits(data, fixed_quota)
    metrics = {
        "dataset": info.name,
        "seed": seed,
        "fixed_quota": fixed_quota,
        "best_epoch": best_epoch,
        "epochs_ran": len(log),
        "validation_accuracy": _accuracy(
            final_logits, data.y, data.val_mask
        ),
        "test_accuracy": _accuracy(final_logits, data.y, data.test_mask),
        "checkpoint_selection": "validation accuracy only",
        "scorer_architecture": "unchanged LocalEdgeScorer",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(output_dir / "training_log.csv", index=False)
    checkpoint = {
        "phase": "final-validation-f1-pretraining",
        "mode": "global_fixed",
        "dataset": info.name,
        "seed": seed,
        "model_state_dict": _cpu_state(best_state),
        "model_config": dict(model_config),
        **metrics,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return checkpoint


def load_dense_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    return {
        key: value.to(device)
        for key, value in checkpoint["model_state_dict"].items()
    }


def load_scorer(
    path: Path,
    in_channels: int,
    model_config: dict[str, Any],
    device: torch.device,
) -> LocalEdgeScorer:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    checkpoint_config = checkpoint.get("model_config", model_config)
    scorer = LocalEdgeScorer(
        in_channels,
        int(checkpoint_config["edge_projection_channels"]),
        int(checkpoint_config["edge_hidden_channels"]),
    ).to(device)
    full_state = checkpoint["model_state_dict"]
    prefix = "edge_scorer."
    state = {
        key[len(prefix) :]: value
        for key, value in full_state.items()
        if key.startswith(prefix)
    }
    if not state:
        state = full_state
    scorer.load_state_dict(state)
    scorer.eval()
    return scorer


def _train_model(
    model: DenseGCN,
    data: Data,
    edge_index: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    training_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    best_validation = -1.0
    best_epoch = 0
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None
    log = []
    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, edge_index)
        loss = F.cross_entropy(
            logits[data.train_mask], data.y[data.train_mask]
        )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(data.x, edge_index)
        validation_accuracy = _accuracy(
            validation_logits, data.y, data.val_mask
        )
        log.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.detach()),
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_accuracy > best_validation + float(
            training_config.get("min_delta", 0.0)
        ):
            best_validation = validation_accuracy
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= int(training_config["patience"]):
            break
    if best_state is None:
        raise RuntimeError("Dense Teacher training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(data.x, edge_index)
    metrics = {
        "best_epoch": best_epoch,
        "epochs_ran": len(log),
        "validation_accuracy": _accuracy(
            final_logits, data.y, data.val_mask
        ),
        "test_accuracy": _accuracy(final_logits, data.y, data.test_mask),
        "checkpoint_selection": "validation accuracy only",
    }
    return best_state, metrics, log


def _gradient_norm(module: nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().pow(2).sum())
    return squared**0.5


def _cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state.items()}


def _accuracy(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> float:
    return float(
        (logits[mask].argmax(dim=-1) == labels[mask]).float().mean()
    )
