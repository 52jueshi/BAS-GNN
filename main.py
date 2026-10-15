"""Train BAS-GNN with a manuscript backbone on one public dataset."""
import argparse
import json


def configuration(args):
    return {
        "model": {
            "backbone": args.backbone,
            "hidden_channels": args.hidden,
            "dropout": args.dropout,
            "gat_heads": args.gat_heads,
            "edge_projection_channels": args.scorer_projection,
            "edge_hidden_channels": args.scorer_hidden,
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "min_delta": 0.0,
            "benchmark_inference": False,
        },
        "student_overrides": {"Cora": {"epochs": 60, "patience": 20}},
        "reference_quota": 8,
        "binary_search_iterations": 80,
        "edge_score_chunk_size": 100000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="CiteSeer",
        choices=[
            "AmazonComputers",
            "AmazonPhoto",
            "CoauthorCS",
            "CoauthorPhysics",
            "CiteSeer",
            "PubMed",
            "Cora",
            "OGBN-Arxiv",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone", choices=["GCN", "GraphSAGE", "GAT"], default="GCN")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output", default="outputs/run")
    parser.add_argument("--scorer", help="Optional existing compatible scorer checkpoint")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--scorer-projection", type=int, default=32)
    parser.add_argument("--scorer-hidden", type=int, default=64)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without loading data or training")
    args = parser.parse_args()
    config = configuration(args)
    if args.dry_run:
        print(json.dumps({"dataset": args.dataset, "seed": args.seed, "device": args.device,
                          "config": config, "stages": ["scorer", "dense checkpoint", "q8 anchor", "CE candidate", "CE+KD candidate", "selected stage2"]}, indent=2))
        return 0
    from bas_gnn.pipeline import run
    print(json.dumps(run(args, config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
