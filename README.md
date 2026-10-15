# BAS-GNN

Reference PyTorch implementation of **BAS-GNN: Budget-Aligned Structured
Sparsification via Global Neighbor-Budget Allocation**.

## Method implemented here

The public runner follows the manuscript protocol:

1. preprocess the candidate graph by removing self-loops, symmetrizing when
   necessary, and coalescing duplicate directed entries;
2. train the shared `LocalEdgeScorer` with hard receiver-wise Top-4 selection
   and a straight-through gradient surrogate;
3. freeze the scorer and construct receiver-wise Softmax importance, rankings,
   realized costs, and Top-q utilities for `Q={4,8,16}`;
4. allocate one quota per receiver with the 80-iteration Lagrangian search under
   the realized Fixed-q8 edge budget, without remainder filling;
5. select a dense checkpoint by validation accuracy, initialize and fine-tune a
   Fixed-q8 anchor from that checkpoint, and copy the complete anchor state into
   both BAS adaptation candidates;
6. train CE and CE+KD (`KL(student || anchor)`, temperature 1, coefficient 1)
   candidates, select by validation accuracy with CE winning exact ties, and
   continue the selected objective with a fresh Adam optimizer;
7. export the static sparse graph and trained target backbone. Deployment
   performs no edge scoring, sorting, allocation, or anchor evaluation.

Candidate self-loops are excluded from the sparsification budget. GCN, mean-
aggregation GraphSAGE, and four-head GAT use their native message-passing rules
on the exported unit-weight topology. The GCN-based scorer-training student is
shared across target backbones; scorer importance is not reused as a propagation
weight.

## Repository layout

| Path | Purpose |
|---|---|
| `main.py` | Single-dataset training and static deployment export |
| `model.py` | Public GCN, GraphSAGE, GAT, and LocalEdgeScorer interface |
| `utils.py` | Data, scoring, allocation, and graph utilities |
| `evaluation.py` | Accuracy and population Macro-F1 |
| `bas_gnn/` | Core BAS-GNN implementation |
| `data/` | Dataset and split notes |
| `result/` | Manuscript-aligned five-seed summaries and released logs |

## Installation

Use Python 3.12. Install the appropriate CPU or CUDA PyTorch build first, then:

```bash
python -m pip install -r requirements.txt
```

The OGB package is included because OGBN-Arxiv is one of the eight manuscript
datasets.

## Usage

```bash
python main.py --help
python main.py --dataset CiteSeer --device cpu --dry-run
python main.py --dataset CiteSeer --seed 0 --device cuda:0 --output outputs/citeseer_s0
python main.py --dataset CiteSeer --backbone GraphSAGE --seed 0 --device cuda:0 --output outputs/citeseer_sage_s0
python main.py --dataset CiteSeer --backbone GAT --seed 0 --device cuda:0 --output outputs/citeseer_gat_s0
python main.py --dataset OGBN-Arxiv --seed 0 --device cuda:0 --output outputs/arxiv_s0
```

For the manuscript's five-seed protocol, run seeds `0,1,2,3,4` with distinct
output directories. Supported names are AmazonComputers, AmazonPhoto,
CoauthorCS, CoauthorPhysics, CiteSeer, Cora, PubMed, and OGBN-Arxiv.

Planetoid datasets use public splits; Amazon and Coauthor use seeded stratified
60/20/20 splits; OGBN-Arxiv uses its official OGB split. Features are normalized
with PyG `NormalizeFeatures`.

`--scorer path/to/checkpoint.pt` reuses a compatible frozen scorer. `--resume`
requires an identical run manifest. Completed stages are reused; an incomplete
stage is not silently accepted.

Outputs include scorer, dense, Fixed-q8 anchor, both stage-1 candidates, stage-2
logs/checkpoints, and `static_model.pt`. The deployment bundle contains the
trained backbone state and identity, exported edge index, selected quota codes,
and budget fields.

## Hyperparameters

- two-layer GCN, mean GraphSAGE, or four-head GAT; hidden width 64 and dropout
  0.5;
- scorer projection width 32 and MLP `130 -> 64 -> 1` with ReLU;
- Adam, learning rate 0.05, weight decay `5e-4`;
- at most 100 epochs and patience 60 for every stage;
- Cora: 60 epochs and patience 20 for every stage;
- scorer quota 4 and straight-through temperature 1;
- quota codebook `{4,8,16}`, Fixed-q8 reference, 80 bisection iterations;
- validation-accuracy checkpoint/objective selection; test metrics only after
  selection;
- five random seeds and population standard deviation for reported results.

## Numerical conventions

Budgets count directed, non-self-loop candidate edges and
`C_i(q)=min(d_i,q)`. Utilities are accumulated in single precision. Local ties
select the smaller quota. The final budget is an upper bound and no post-search
edge filling is performed.
