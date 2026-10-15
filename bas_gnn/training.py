"""Protocol-matched, instrumented two-layer backbone training."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
import torch
from torch.nn import functional as F

from .graphs import WeightedStaticGraph
from .backbones import build_backbone
from .runtime import set_seed


OBJECTIVE_CE = "CE"
OBJECTIVE_KD = "CE+KD"


def train_or_load_gcn(
    context: Any,
    graph: WeightedStaticGraph,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    *,
    phase: str,
    method: str,
    initial_checkpoint: Path | None = None,
    objective: str = OBJECTIVE_CE,
    teacher_logits: torch.Tensor | None = None,
) -> tuple[dict[str, Any], Path]:
    """Train one immutable-graph model and select solely by validation accuracy."""
    checkpoint_path = output_dir / "best_model.pt"
    metrics_path = output_dir / "metrics.json"
    if checkpoint_path.exists() and metrics_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        _validate_cached(
            checkpoint, phase, method, graph, objective, model_config, device
        )
        return json.loads(metrics_path.read_text(encoding="utf-8")), checkpoint_path
    if checkpoint_path.exists() or metrics_path.exists():
        raise RuntimeError(f"partial F2C training artifacts: {output_dir}")
    if objective not in {OBJECTIVE_CE, OBJECTIVE_KD}:
        raise ValueError(f"unsupported objective: {objective}")
    if objective == OBJECTIVE_KD and teacher_logits is None:
        raise ValueError("CE+KD requires frozen q8 teacher logits")

    set_seed(int(context.seed))
    model = build_backbone(
        model_config, context.info.num_features, context.info.num_classes
    ).to(device)
    if initial_checkpoint is not None:
        initial = torch.load(initial_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(initial["model_state_dict"])

    data = context.data
    edges = graph.edge_index.to(device).clone()
    weights = (
        None
        if graph.edge_weight is None
        else graph.edge_weight.to(device).clone()
    )
    immutable_edges = edges.clone()
    immutable_weights = None if weights is None else weights.clone()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    teacher_log_probability = None
    if teacher_logits is not None:
        teacher_log_probability = F.log_softmax(teacher_logits.detach(), dim=-1)

    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    best_validation = -1.0
    best_epoch = 0
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None
    log: list[dict[str, float | int]] = []
    for epoch in range(1, int(training_config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, edges, edge_weight=weights)
        ce = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        kd = logits.new_zeros(())
        if objective == OBJECTIVE_KD:
            student_log_probability = F.log_softmax(logits[data.train_mask], dim=-1)
            student_probability = student_log_probability.exp()
            kd = (
                student_probability
                * (
                    student_log_probability
                    - teacher_log_probability[data.train_mask]
                )
            ).sum(dim=-1).mean()
        loss = ce + kd
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(data.x, edges, edge_weight=weights)
        validation_accuracy = _accuracy(
            validation_logits, data.y, data.val_mask
        )
        log.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "ce": float(ce.detach()),
                "kd_student_to_teacher": float(kd.detach()),
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
    _synchronize(device)
    training_seconds = time.perf_counter() - started
    peak_memory_mib = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
        if device.type == "cuda"
        else float("nan")
    )
    if best_state is None:
        raise RuntimeError("F2C training produced no validation checkpoint")
    if not torch.equal(edges, immutable_edges):
        raise RuntimeError("static graph changed during F2C training")
    if weights is not None and not torch.equal(weights, immutable_weights):
        raise RuntimeError("static edge weights changed during F2C training")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(data.x, edges, edge_weight=weights)
    inference_ms = (
        benchmark_inference_ms(model, data.x, edges, weights, device)
        if bool(training_config.get("benchmark_inference", False))
        else None
    )
    metrics: dict[str, Any] = {
        "method": method,
        "backbone": str(model_config.get("backbone", "GCN")),
        "seed": int(context.seed),
        "best_epoch": best_epoch,
        "epochs_ran": len(log),
        "validation_accuracy": _accuracy(final_logits, data.y, data.val_mask),
        "validation_macro_f1": _macro_f1(final_logits, data.y, data.val_mask),
        "test_accuracy": _accuracy(final_logits, data.y, data.test_mask),
        "test_macro_f1": _macro_f1(final_logits, data.y, data.test_mask),
        "optimizer": "Adam",
        "learning_rate": float(training_config["learning_rate"]),
        "weight_decay": float(training_config["weight_decay"]),
        "max_epochs": int(training_config["epochs"]),
        "patience": int(training_config["patience"]),
        "checkpoint_selection": "validation accuracy only",
        "test_labels_used_for_training_or_selection": False,
        "objective": objective,
        "kd_weight": 1.0 if objective == OBJECTIVE_KD else 0.0,
        "training_seconds": training_seconds,
        "inference_ms": inference_ms,
        "peak_memory_mib": peak_memory_mib,
        "initialization_checkpoint": (
            None if initial_checkpoint is None else str(initial_checkpoint.resolve())
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(output_dir / "training_log.csv", index=False)
    torch.save(
        {
            "phase": phase,
            "method": method,
            "backbone": str(model_config.get("backbone", "GCN")),
            "objective": objective,
            "seed": int(context.seed),
            "model_state_dict": _cpu_state(best_state),
            "sparse_edge_index": edges.detach().cpu(),
            "edge_weight": None if weights is None else weights.detach().cpu(),
            "initialization_checkpoint": metrics["initialization_checkpoint"],
            "checkpoint_selection": "validation accuracy only",
            "test_labels_used_for_training_or_selection": False,
        },
        checkpoint_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, checkpoint_path


@torch.no_grad()
def checkpoint_logits(
    context: Any,
    checkpoint_path: Path,
    graph: WeightedStaticGraph,
    model_config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_backbone(
        model_config, context.info.num_features, context.info.num_classes
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    weights = None if graph.edge_weight is None else graph.edge_weight.to(device)
    return model(context.data.x, graph.edge_index.to(device), edge_weight=weights)


def benchmark_inference_ms(
    model: torch.nn.Module,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    device: torch.device,
    *,
    warmup: int = 30,
    iterations: int = 200,
    repetitions: int = 5,
) -> float:
    """Mean latency for the manuscript's five repeated synchronized trials."""
    if warmup < 0 or iterations <= 0 or repetitions <= 0:
        raise ValueError("warmup must be non-negative; iterations and repetitions positive")
    model.eval()
    measurements = []
    with torch.no_grad():
        for _ in range(repetitions):
            for _ in range(warmup):
                model(features, edge_index, edge_weight=edge_weight)
            _synchronize(device)
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    model(features, edge_index, edge_weight=edge_weight)
                end.record()
                end.synchronize()
                measurements.append(float(start.elapsed_time(end)) / iterations)
            else:
                started = time.perf_counter()
                for _ in range(iterations):
                    model(features, edge_index, edge_weight=edge_weight)
                measurements.append(
                    (time.perf_counter() - started) * 1000.0 / iterations
                )
    return sum(measurements) / len(measurements)


def _validate_cached(
    checkpoint: dict[str, Any],
    phase: str,
    method: str,
    graph: WeightedStaticGraph,
    objective: str,
    model_config: dict[str, Any],
    device: torch.device,
) -> None:
    if checkpoint.get("phase") != phase or checkpoint.get("method") != method:
        raise RuntimeError("cached F2C checkpoint provenance mismatch")
    if checkpoint.get("objective") != objective:
        raise RuntimeError("cached F2C objective mismatch")
    expected_backbone = str(model_config.get("backbone", "GCN"))
    if checkpoint.get("backbone", "GCN") != expected_backbone:
        raise RuntimeError("cached F2C backbone mismatch")
    if not torch.equal(
        checkpoint["sparse_edge_index"].to(device), graph.edge_index.to(device)
    ):
        raise RuntimeError("cached F2C graph mismatch")
    stored_weight = checkpoint.get("edge_weight")
    if graph.edge_weight is None:
        if stored_weight is not None:
            raise RuntimeError("cached F2C edge-weight mismatch")
    elif stored_weight is None or not torch.allclose(
        stored_weight.to(device), graph.edge_weight.to(device)
    ):
        raise RuntimeError("cached F2C edge-weight mismatch")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    return float((logits[mask].argmax(dim=-1) == labels[mask]).float().mean())


def _macro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
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
        values.append(
            torch.where(
                denominator > 0,
                2 * true_positive / denominator,
                torch.zeros_like(denominator),
            )
        )
    return float(torch.stack(values).mean())


def _cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state.items()}
