"""Public model interface for BAS-GNN."""

from bas_gnn.gcn import DenseGCN
from bas_gnn.backbones import DenseGAT, DenseGraphSAGE
from bas_gnn.scorer import LocalEdgeScorer

__all__ = ["DenseGCN", "DenseGraphSAGE", "DenseGAT", "LocalEdgeScorer"]
