"""Single-dataset scorer, same-backbone anchor, and BAS adaptation pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.utils import coalesce, is_undirected, softmax

from .datasets import load_node_classification_data
from .graphs import WeightedStaticGraph, build_fixed_graph
from .pretraining import train_local_edge_scorer, load_scorer
from .scoring import local_edge_scores, rank_edges_by_receiver
from .structured import allocate, build_graph
from .training import train_or_load_gcn, checkpoint_logits

DATASETS = {"AmazonComputers": "Computers", "AmazonPhoto": "Photo",
            "CoauthorCS": "CS", "CoauthorPhysics": "Physics",
            "CiteSeer": "CiteSeer", "Cora": "Cora", "PubMed": "PubMed",
            "OGBN-Arxiv": "ogbn-arxiv"}


def make_context(data, info, scorer, config, seed):
    candidate = coalesce(data.edge_index, num_nodes=data.num_nodes)
    if candidate.size(1) == 0 or bool((candidate[0] == candidate[1]).any()):
        raise ValueError("Use a nonempty, loop-free candidate graph; GCN adds self-loops")
    if not is_undirected(candidate, num_nodes=data.num_nodes):
        raise ValueError("The standard GCN protocol expects undirected candidate edges")
    degree = torch.bincount(candidate[1], minlength=data.num_nodes)
    scores = local_edge_scores(scorer, data.x, candidate, degree, config["edge_score_chunk_size"])
    curve = SimpleNamespace(edge_attention=softmax(scores, candidate[1], num_nodes=data.num_nodes),
                            edge_rank=rank_edges_by_receiver(candidate[1], scores))
    return SimpleNamespace(data=data, info=info, seed=seed, candidate_edge_index=candidate,
                           degree=degree, curve=curve)


def adapt(context, graph, fixed_graph, anchor, config, device, output):
    """Two stage adaptation; stage 1 compares CE and CE+KL(student||teacher)."""
    teacher = checkpoint_logits(context, anchor, fixed_graph, config["model"], device)
    candidates = []
    for objective, folder in (("CE", "stage1_ce"), ("CE+KD", "stage1_ce_kd")):
        metrics, checkpoint = train_or_load_gcn(
            context, graph, config["model"], config["training"], device, output / folder,
            phase="bas-core-stage1", method="Structured-Importance BAS", initial_checkpoint=anchor,
            objective=objective, teacher_logits=teacher)
        candidates.append((objective, metrics, checkpoint))
    # max returns the first maximum, preserving CE preference on an exact tie.
    objective, _, selected = max(candidates, key=lambda item: item[1]["validation_accuracy"])
    return train_or_load_gcn(
        context, graph, config["model"], config["training"], device, output / "stage2",
        phase="bas-core-stage2", method="Structured-Importance BAS", initial_checkpoint=selected,
        objective=objective, teacher_logits=teacher)


def run(args, config):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a CUDA PyTorch build or use --device cpu")
    if config["reference_quota"] != 8:
        raise ValueError("This core release uses the frozen q8 anchor protocol")
    backbone = str(config["model"].get("backbone", "GCN"))
    output = Path(args.output)
    signature = {"dataset": args.dataset, "seed": args.seed, "config": config,
                 "device": str(device), "scorer_sha256": None}
    if args.scorer:
        if not Path(args.scorer).is_file():
            raise FileNotFoundError(f"Requested scorer does not exist: {args.scorer}")
        signature["scorer_sha256"] = hashlib.sha256(Path(args.scorer).read_bytes()).hexdigest()
    manifest = output / "run_config.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume or not manifest.exists():
            raise RuntimeError("Output is not empty. Use a fresh --output or --resume with its manifest")
        if json.loads(manifest.read_text(encoding="utf-8")) != signature:
            raise RuntimeError("Resume configuration mismatch; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    data, info = load_node_classification_data({
        "name": DATASETS[args.dataset], "dataset_id": args.dataset,
        "root": args.data_root, "split_root": str(output / "splits"),
        "normalize_features": True, "persist_split": True, "split": "public",
        "train_ratio": 0.6, "val_ratio": 0.2}, args.seed)
    data = data.to(device)
    train_config = dict(config["training"])
    train_config.update(config.get("student_overrides", {}).get(args.dataset, {}))
    effective_config = {**config, "training": train_config}
    scorer_path = Path(args.scorer) if args.scorer else output / "scorer" / "best_model.pt"
    if not scorer_path.exists():
        train_local_edge_scorer(data, info, config["model"], train_config,
                               device, args.seed, scorer_path.parent)
    scorer = load_scorer(scorer_path, info.num_features, config["model"], device)
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    context = make_context(data, info, scorer, config, args.seed)
    fixed = build_fixed_graph(context, 8)
    dense = WeightedStaticGraph(
        context.candidate_edge_index,
        None,
        context.degree,
        {},
        0.0,
    )
    _, dense_checkpoint = train_or_load_gcn(
        context,
        dense,
        config["model"],
        train_config,
        device,
        output / "dense",
        phase="bas-dense-checkpoint",
        method=f"Dense {backbone}",
    )
    _, anchor = train_or_load_gcn(
        context,
        fixed,
        config["model"],
        train_config,
        device,
        output / "anchor",
        phase="bas-core-anchor",
        method=f"Fixed-q8 {backbone}",
        initial_checkpoint=dense_checkpoint,
    )
    allocation = allocate(context, fixed.edge_index.size(1), config["binary_search_iterations"])
    graph = build_graph(context, allocation)
    metrics, checkpoint = adapt(context, graph, fixed, anchor, effective_config, device, output)
    final = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.save({"model_state_dict": final["model_state_dict"],
                "model_config": config["model"], "num_features": info.num_features,
                "backbone": backbone,
                "num_classes": info.num_classes, "num_nodes": info.num_nodes,
                "edge_index": graph.edge_index.cpu(), "edge_weight": graph.edge_weight,
                "quota_codes": allocation.selected_quota_codes.cpu(),
                "budget_cap": allocation.target_budget, "actual_edges": allocation.actual_budget},
               output / "static_model.pt")
    return {"checkpoint": str(checkpoint), "deployment_bundle": str(output / "static_model.pt"),
            "validation_accuracy": metrics["validation_accuracy"],
            "validation_macro_f1": metrics["validation_macro_f1"],
            "test_accuracy": metrics["test_accuracy"],
            "test_macro_f1": metrics["test_macro_f1"], "objective": metrics["objective"],
            "budget_cap": allocation.target_budget, "actual_edges": allocation.actual_budget}
