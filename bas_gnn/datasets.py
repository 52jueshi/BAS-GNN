"""Lazy dataset interfaces; only the selected dataset is downloaded."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.datasets import Actor, Amazon, Coauthor, Planetoid, WebKB, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata needed to construct a node-classification model."""

    name: str
    num_features: int
    num_classes: int
    num_nodes: int
    num_edges: int


PLANETOID = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
AMAZON = {"computers": "Computers", "photo": "Photo"}
COAUTHOR = {"cs": "CS", "physics": "Physics"}
WEBKB = {"cornell": "Cornell", "texas": "Texas", "wisconsin": "Wisconsin"}
WIKIPEDIA = {"chameleon": "chameleon", "squirrel": "squirrel"}


def load_node_classification_data(
    data_config: dict[str, Any], seed: int
) -> tuple[Data, DatasetInfo]:
    """Load one configured dataset without touching any other dataset.

    Supported names are Cora, CiteSeer, PubMed, Computers, Photo, CS,
    Physics, Actor, Cornell, Texas, Wisconsin, Chameleon, Squirrel,
    ogbn-arxiv, and ogbn-products. OGB support is lazy and requires the
    optional ``ogb`` package.
    """
    name = str(data_config["name"])
    dataset_id = str(data_config.get("dataset_id", name))
    key = name.lower()
    root = Path(data_config.get("root", "datasets")).resolve()
    transform = NormalizeFeatures() if data_config.get("normalize_features", True) else None

    if key in PLANETOID:
        dataset = Planetoid(
            root=str(root / "Planetoid"),
            name=PLANETOID[key],
            split=str(data_config.get("split", "public")),
            transform=transform,
        )
        data = dataset[0]
        num_classes = dataset.num_classes
    elif key in AMAZON:
        dataset = Amazon(root=str(root / "Amazon"), name=AMAZON[key], transform=transform)
        data = dataset[0]
        num_classes = dataset.num_classes
        _ensure_masks(data, seed, data_config, f"Amazon{AMAZON[key]}")
    elif key in COAUTHOR:
        dataset = Coauthor(root=str(root / "Coauthor"), name=COAUTHOR[key], transform=transform)
        data = dataset[0]
        num_classes = dataset.num_classes
        _ensure_masks(data, seed, data_config, f"Coauthor{COAUTHOR[key]}")
    elif key == "actor":
        dataset = Actor(root=str(root / "Actor"), transform=transform)
        data = dataset[0]
        num_classes = dataset.num_classes
        _select_mask_column(data, int(data_config.get("split_index", 0)))
    elif key in WEBKB:
        dataset = WebKB(root=str(root / "WebKB"), name=WEBKB[key], transform=transform)
        data = dataset[0]
        num_classes = dataset.num_classes
        _select_mask_column(data, int(data_config.get("split_index", 0)))
    elif key in WIKIPEDIA:
        dataset = WikipediaNetwork(
            root=str(root / "WikipediaNetwork"), name=WIKIPEDIA[key], transform=transform
        )
        data = dataset[0]
        num_classes = dataset.num_classes
        _select_mask_column(data, int(data_config.get("split_index", 0)))
    elif key in {"ogbn-arxiv", "ogbn-products"}:
        data, num_classes = _load_ogb(key, root, transform)
    else:
        supported = sorted(
            set(PLANETOID) | set(AMAZON) | set(COAUTHOR) | {"actor"}
            | set(WEBKB) | set(WIKIPEDIA) | {"ogbn-arxiv", "ogbn-products"}
        )
        raise ValueError(f"Unsupported dataset {name!r}. Supported names: {supported}")

    # The manuscript defines a common candidate graph: no self-loops,
    # symmetrized where necessary, and duplicate directed entries coalesced.
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
    data.edge_index = coalesce(edge_index, num_nodes=data.num_nodes)
    _validate_data(data)
    info = DatasetInfo(
        name=dataset_id,
        num_features=int(data.num_features),
        num_classes=int(num_classes),
        num_nodes=int(data.num_nodes),
        num_edges=int(data.edge_index.size(1)),
    )
    return data, info


def _ensure_masks(
    data: Data, seed: int, config: dict[str, Any], dataset_id: str
) -> None:
    if all(hasattr(data, name) for name in ("train_mask", "val_mask", "test_mask")):
        return
    train_ratio = float(config.get("train_ratio", 0.6))
    val_ratio = float(config.get("val_ratio", 0.2))
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than one")
    labels = data.y.cpu().numpy()
    split_path = Path(config.get("split_root", "splits")) / dataset_id / f"seed_{seed}.pt"
    label_checksum = hashlib.sha256(labels.tobytes()).hexdigest()
    if bool(config.get("persist_split", True)) and split_path.exists():
        payload = torch.load(split_path, map_location="cpu", weights_only=True)
        _validate_split_payload(payload, dataset_id, seed, data.num_nodes, label_checksum)
        data.train_mask = _index_mask(payload["train_idx"].numpy(), data.num_nodes)
        data.val_mask = _index_mask(payload["val_idx"].numpy(), data.num_nodes)
        data.test_mask = _index_mask(payload["test_idx"].numpy(), data.num_nodes)
        data.split_path = str(split_path.resolve())
        return
    indices = np.arange(data.num_nodes)
    train_idx, remaining = train_test_split(
        indices, train_size=train_ratio, random_state=seed, stratify=labels
    )
    relative_val = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        remaining,
        train_size=relative_val,
        random_state=seed,
        stratify=labels[remaining],
    )
    data.train_mask = _index_mask(train_idx, data.num_nodes)
    data.val_mask = _index_mask(val_idx, data.num_nodes)
    data.test_mask = _index_mask(test_idx, data.num_nodes)
    if bool(config.get("persist_split", True)):
        split_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": dataset_id,
            "seed": int(seed),
            "num_nodes": int(data.num_nodes),
            "label_checksum": label_checksum,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "train_idx": torch.as_tensor(train_idx, dtype=torch.long),
            "val_idx": torch.as_tensor(val_idx, dtype=torch.long),
            "test_idx": torch.as_tensor(test_idx, dtype=torch.long),
        }
        torch.save(payload, split_path)
        data.split_path = str(split_path.resolve())


def _select_mask_column(data: Data, split_index: int) -> None:
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, name)
        if mask.dim() == 2:
            if not 0 <= split_index < mask.size(1):
                raise ValueError(f"split_index={split_index} is out of range for {name}")
            setattr(data, name, mask[:, split_index])


def _index_mask(indices: np.ndarray, size: int) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool)
    mask[torch.as_tensor(indices, dtype=torch.long)] = True
    return mask


def _validate_split_payload(
    payload: dict[str, Any],
    dataset_id: str,
    seed: int,
    num_nodes: int,
    label_checksum: str,
) -> None:
    expected = {
        "dataset": dataset_id,
        "seed": int(seed),
        "num_nodes": int(num_nodes),
        "label_checksum": label_checksum,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Persisted split mismatch for {key}: {payload.get(key)!r} != {value!r}")
    parts = [payload[name].to(torch.long) for name in ("train_idx", "val_idx", "test_idx")]
    combined = torch.cat(parts)
    if combined.numel() != num_nodes or torch.unique(combined).numel() != num_nodes:
        raise ValueError("Persisted split indices must be disjoint and cover every node exactly once")
    if combined.numel() and (int(combined.min()) < 0 or int(combined.max()) >= num_nodes):
        raise ValueError("Persisted split contains an out-of-range node index")


def _load_ogb(key: str, root: Path, transform: Any) -> tuple[Data, int]:
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:
        raise ImportError("OGB datasets require `pip install ogb`") from exc
    dataset = PygNodePropPredDataset(name=key, root=str(root / "OGB"), transform=transform)
    data = dataset[0]
    data.y = data.y.view(-1)
    split = dataset.get_idx_split()
    data.train_mask = _index_mask(split["train"].cpu().numpy(), data.num_nodes)
    data.val_mask = _index_mask(split["valid"].cpu().numpy(), data.num_nodes)
    data.test_mask = _index_mask(split["test"].cpu().numpy(), data.num_nodes)
    return data, int(dataset.num_classes)


def _validate_data(data: Data) -> None:
    if data.x is None or data.y is None:
        raise ValueError("Node-classification data must provide x and y")
    if data.edge_index.dtype != torch.long or data.edge_index.shape[0] != 2:
        raise TypeError("edge_index must have shape [2, E] and dtype torch.long")
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, name, None)
        if mask is None or mask.dtype != torch.bool or mask.shape != (data.num_nodes,):
            raise ValueError(f"{name} must be a bool tensor with shape [num_nodes]")
