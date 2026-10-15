# Data

Dataset files are not redistributed. `main.py` uses public PyTorch Geometric
loaders for Cora, CiteSeer, PubMed, Amazon Computers, Amazon Photo, Coauthor CS,
and Coauthor Physics, plus the official OGB loader for OGBN-Arxiv. Files are
downloaded into the path supplied by `--data-root` and remain subject to their
original licenses and terms.

Common graph preprocessing removes self-loops, symmetrizes graphs when
applicable, and coalesces duplicate directed entries before the candidate edge
set and Fixed-q8 budget are formed.

Planetoid datasets use their public splits. Amazon and Coauthor datasets use
the seeded stratified 60/20/20 split implemented in `bas_gnn/datasets.py`.
OGBN-Arxiv uses the official OGB split. Manuscript results use seeds
`0,1,2,3,4` and report population standard deviation.
