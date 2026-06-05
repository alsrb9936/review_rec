# UTILS KNOWLEDGE BASE

**Generated:** 2026-06-03

## OVERVIEW
Shared helpers for seeding, statistics derivation, graph construction, metrics, and preprocessing (GloVe / BERT).

## STRUCTURE
```
utils/
├── utils.py              # Seed, stats, dataloader, graph builders
├── load_data.py          # Raw interaction loading + train/valid/test split
├── metric.py             # RMSE, MSE, MAE
├── glove_pro.py          # GloVe tokenization + .npy generation
└── bert_pro.py           # BERT whitening embedding + .npy generation
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add metric | `metric.py` | Only regression metrics currently |
| Fix cold-start split | `load_data.py:split_by_ratio` | Moves unseen users/items from valid/test to train |
| LightGCN adjacency | `utils.py:build_lightgcn_norm_adj_from_train` | Sparse coo normalized symmetrically |
| RGCL graph | `utils.py:build_rgcl_graph_from_train` | DGL heterograph with rating edge types + review_feat |
| Data type mapping | `utils.py` | `GLOVE_MODEL_NAMES` / `BERT_MODEL_NAMES` |

## CONVENTIONS
- `utils/__init__.py` is empty; import directly (`from utils.utils import ...`).
- `set_stats_from_npy` derives `num_users`/`num_items` by scanning split `.npy` files.
- `get_dataloader` instantiates datasets and returns train/valid/test loaders.

## UNIQUE STYLES
- **Percentile capping**: `glove_pro.py` uses percentile-based max_len / max_review_count instead of absolute max.
- **BERT whitening**: `bert_pro.py` computes SVD-whitened BERT embeddings for dimensionality reduction.
