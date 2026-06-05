# TRAINER KNOWLEDGE BASE

**Generated:** 2026-06-03

## OVERVIEW
`BaseTrainer` defines the training loop; each model has a derived trainer implementing `train_step`, `evaluate`, and `get_metric_name`.

## STRUCTURE
```
trainer/
├── __init__.py           # MODEL_TRAINER_DICT registry
├── base_trainer.py       # Abstract: loop, early stop, checkpoint, log
├── neumf_trainer.py
├── deepconn_trainer.py
├── narre_trainer.py
├── daml_trainer.py
├── transnet_trainer.py
├── lightgcn_trainer.py
└── rgcl_trainer.py
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Change early stopping | `base_trainer.py` | `early_stop_patience`, eval every `eval_step` |
| Change optimizer | `base_trainer.__init__` | Adam or SGD; lr scheduler is `ExponentialLR` |
| Add metric | `base_trainer.train` + derived `evaluate` | Currently logs RMSE/MSE/MAE |
| Device batch move | `base_trainer._move_batch_to_device` | Dict-level tensor `.to(device)` |

## CONVENTIONS
- Trainers call `model.calculate_loss(user_id, item_id, rating)` in `train_step`.
- `evaluate` collects `all_preds`/`all_targets` and returns `compute_all_metrics()`.
- `get_metric_name` returns the key used for early stopping (usually `"rmse"`).

## UNIQUE STYLES
- **Lower-is-better metric**: `current_metric < best_metric_value` triggers checkpoint save; no support for higher-is-better currently.
