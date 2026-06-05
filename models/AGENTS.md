# MODELS KNOWLEDGE BASE

**Generated:** 2026-06-03

## OVERVIEW
7 recommendation model implementations registered in `__init__.py`.

## STRUCTURE
```
models/
├── __init__.py       # MODEL_DICT registry
├── neumf.py          # Matrix factorization + MLP fusion
├── deepconn.py       # Dual CNN over user/item review text
├── narre.py          # Neural attentive review recommendation
├── daml.py           # Dual attention matching layer
├── transnet.py       # TransformNet with text CNN + FM
├── lightgcn.py       # Graph convolution, no review text
└── rgcl.py           # Review-aware graph contrastive learning (DGL)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add model | `__init__.py` + new `{name}.py` | Register class in `MODEL_DICT` |
| Graph conv review | `rgcl.py` | `ReviewAwareGraphConv`, `RGCLGraphEncoder` |
| Sparse adj model | `lightgcn.py` | Accepts `norm_adj` tensor in constructor |
| Text CNN encoders | `deepconn.py`, `transnet.py` | `CNN`, `TextCNNEncoder` |

## CONVENTIONS
- All models inherit `nn.Module` and accept `cfg: DictConfig`.
- LightGCN uniquely accepts an extra `norm_adj: torch.Tensor` arg.
- RGCL depends on `dgl` and uses heterogeneous graphs.
- `_init_weights()` is common but not enforced by base class.

## ANTI-PATTERNS
- `models/__init__.py` imports `RGCL` twice (lines 4 and 8).

## UNIQUE STYLES
- **LightGCN constructor override**: `run.py` passes `norm_adj` only for this model.
- **RGCL**: Full-batch graph model with `ContrastLoss` and rating-specific edge types.
