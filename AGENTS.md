# PROJECT KNOWLEDGE BASE

**Generated:** 2026-06-03
**Commit:** a6bf5ea
**Branch:** main

## OVERVIEW
Reproducibility study of 7 review-aware recommendation models (NeuMF, DeepCoNN, NARRE, RGCL, DAML, LightGCN, TransNet) on Amazon review datasets. PyTorch + Hydra configuration management.

## STRUCTURE
```
./
├── configs/          # Hydra configs: base + per-model overrides
├── data/             # Dataset classes (1 per model)
├── dataset/          # Preprocessed .npy artifacts (glove | bert)
├── models/           # Model implementations (nn.Module registry)
├── outputs/          # Training artifacts: checkpoints + logs + results
├── scripts/          # GPU-distributed run scripts + preprocess
├── trainer/          # Trainer classes (BaseTrainer + 7 derived)
├── utils/            # Shared: metrics, graph builders, preprocessors
├── preprocess.py     # Entry: raw data -> train/valid/test .npy
└── run.py            # Entry: config -> train -> evaluate
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add new model | `models/`, `data/`, `trainer/`, `configs/model/` | Must register in `*_DICT` of each package `__init__.py` |
| Change hyperparams | `configs/model/{model}.yaml` | Overrides `configs/config.yaml` defaults |
| Fix data paths | `preprocess.py` lines 87-93 | Hardcoded absolute paths to dataset + word2vec |
| Change split ratio | `utils/load_data.py` `split_by_ratio` | Default 80/10/10; handles cold-start by moving to train |
| Review preprocessing | `utils/glove_pro.py` or `utils/bert_pro.py` | GloVe tokenizes; BERT uses whitening embeddings |
| Add metric | `utils/metric.py` | RMSE/MSE/MAE only currently |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `run.py:main` | Function | Entry | Hydra main: loads config, instantiates model/trainer/data, trains |
| `MODEL_DICT` | Dict | `models/__init__.py` | Registry: str -> nn.Module class |
| `DATASET_DICT` | Dict | `data/__init__.py` | Registry: str -> Dataset class |
| `MODEL_TRAINER_DICT` | Dict | `trainer/__init__.py` | Registry: str -> Trainer class |
| `BaseTrainer` | Class | `trainer/base_trainer.py` | Abstract: training loop, early stop, checkpoint, logging |
| `set_stats_from_npy` | Function | `utils/utils.py` | Derives `num_users`/`num_items` from .npy files |
| `build_lightgcn_norm_adj_from_train` | Function | `utils/utils.py` | Sparse normalized adjacency for LightGCN |
| `build_rgcl_graph_from_train` | Function | `utils/utils.py` | DGL heterogeneous graph with rating edges + review features |
| `split_by_ratio` | Function | `utils/load_data.py` | Splits + cold-start guard |

## CONVENTIONS
- **Registry pattern**: Every model must be registered in `models/__init__.py`, `data/__init__.py`, `trainer/__init__.py`.
- **Model triad**: Each model gets `models/{name}.py`, `data/{name}_dataset.py`, `trainer/{name}_trainer.py`, `configs/model/{name}.yaml`.
- **Config overrides**: Hydra composes `configs/config.yaml` + `configs/model/{model}.yaml` + CLI args. Use `omegaconf.open_dict` for runtime mutations.
- **Data types**: `glove` (token id sequences) or `bert` (whitened review embeddings). `utils/utils.py` maps model names to types via `GLOVE_MODEL_NAMES` / `BERT_MODEL_NAMES`.
- **Checkpointing**: `BaseTrainer` saves `{model_name}_best.pt` with state_dict + optimizer + epoch + best_metric.

## ANTI-PATTERNS (THIS PROJECT)
- Hardcoded absolute paths in `preprocess.py` (dataset root, glove file, stopwords, punctuations). Must edit for new environments.
- `utils/__init__.py` is empty; utilities imported via direct module paths (e.g., `from utils.utils import ...`).
- No tests anywhere.
- `models/__init__.py` imports `RGCL` twice (lines 4 and 8).

## UNIQUE STYLES
- **RGCL full-batch**: `get_dataloader` sets `batch_size=len(train_dataset)` and `shuffle=False` only for RGCL.
- **LightGCN sparse adj**: Normalized sparse coo tensor built once in `run.py` and passed to model constructor.
- **Cold-start fix**: `split_by_ratio` moves valid/test rows with unseen users/items back to train.
- **Review text models**: DeepCoNN, NARRE, TransNet, DAML require review text; NeuMF and LightGCN do not.

## COMMANDS
```bash
# Preprocess (hardcoded paths — edit preprocess.py first)
python preprocess.py data.dataset=Amazon_Musical_Instruments_14 data.type=glove
python preprocess.py data.dataset=Amazon_Musical_Instruments_14 data.type=bert experiment.device=0

# Train single model
python run.py model=neumf data.dataset=Amazon_Musical_Instruments_14 experiment.seed=42 experiment.device=0

# Batch runs (see scripts/)
bash scripts/gpu_0.sh
bash scripts/preprocess.sh
```

## NOTES
- `preprocess.py` unconditionally overwrites `cfg.data.data_root` and other paths at runtime. Review before running in a new environment.
- BERT preprocessing requires GPU (`experiment.device`) and downloads `transformers` models on first run.
- `dataset/` contains subdirs per dataset per embedding type (e.g., `dataset/Amazon_Beauty_18/glove/`).
- Outputs are timestamped: `{model_name}_{dataset}_{seed}_{YYYYMMDD_HHMMSS}/`.
