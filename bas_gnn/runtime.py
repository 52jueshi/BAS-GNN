"""Runtime, reproducibility, and device helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda config to a concrete torch device."""
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={requested!r} was requested, but this PyTorch build has no usable CUDA device"
        )
    return device


def linear_schedule(start: float, end: float, epoch: int, total_epochs: int) -> float:
    """Linearly interpolate from start to end over all epochs."""
    if total_epochs <= 1:
        return float(end)
    ratio = min(max(epoch / (total_epochs - 1), 0.0), 1.0)
    return float(start + ratio * (end - start))
