"""Node-classification metrics used by the compact public interface."""

from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    _validate(logits, labels, mask)
    return float((logits[mask].argmax(dim=-1) == labels[mask]).float().mean())


@torch.no_grad()
def macro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """Population Macro-F1 over classes represented by the model output."""
    _validate(logits, labels, mask)
    prediction = logits[mask].argmax(dim=-1)
    target = labels[mask]
    values = []
    for class_id in range(logits.size(1)):
        predicted = prediction == class_id
        actual = target == class_id
        true_positive = (predicted & actual).sum().to(torch.float64)
        false_positive = (predicted & ~actual).sum().to(torch.float64)
        false_negative = (~predicted & actual).sum().to(torch.float64)
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(torch.where(denominator > 0, 2 * true_positive / denominator, 0.0))
    return float(torch.stack(values).mean())


@torch.no_grad()
def evaluate(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    return {"accuracy": accuracy(logits, labels, mask), "macro_f1": macro_f1(logits, labels, mask)}


def _validate(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> None:
    if logits.dim() != 2 or labels.shape != (logits.size(0),):
        raise ValueError("Expected logits [N,C] and labels [N]")
    if mask.dtype != torch.bool or mask.shape != (logits.size(0),) or not bool(mask.any()):
        raise ValueError("mask must be a nonempty boolean tensor with shape [N]")
