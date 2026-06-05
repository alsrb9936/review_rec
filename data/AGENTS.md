# DATA KNOWLEDGE BASE

**Generated:** 2026-06-03

## OVERVIEW
7 `torch.utils.data.Dataset` subclasses, one per model, loading `.npy` arrays produced by `preprocess.py`.

## STRUCTURE
```
data/
├── __init__.py           # DATASET_DICT registry
├── neumf_dataset.py
├── deepconn_dataset.py
├── narre_dataset.py
├── daml_dataset.py
├── transnet_dataset.py
├── lightgcn_dataset.py
└── rgcl_dataset.py
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add dataset | `__init__.py` + new `{name}_dataset.py` | Register in `DATASET_DICT` |
| Review text loaders | `deepconn_dataset.py`, `narre_dataset.py`, etc. | Load `*_review.npy` or `*_review_text.npy` |
| Graph dataset | `rgcl_dataset.py` | Uses DGL heterogeneous graph from `utils` |

## CONVENTIONS
- Constructor signature: `(cfg: DictConfig, split: str = "train")`.
- `_load_data(split)` reads `dataset/{dataset}/{type}/{split}_*.npy`.
- `__getitem__` returns a dict with tensor values.
- Review-text models load additional `.npy` files for token sequences or embeddings.

## UNIQUE STYLES
- **Data type routing**: `cfg.data.type` is either `glove` or `bert`; some models override this in `utils/utils.py`.
