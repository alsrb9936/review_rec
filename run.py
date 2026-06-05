import os
import json
from collections.abc import Sized
from datetime import datetime
from typing import Protocol, cast

import hydra
import numpy as np
from numpy.typing import NDArray

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, Subset

from data import DATASET_DICT
from models import MODEL_DICT
from trainer import MODEL_TRAINER_DICT
from utils.metric import compute_all_metrics
from utils.utils import set_seed, set_stats_from_npy, get_dataloader, build_lightgcn_norm_adj_from_train, build_recafr_norm_adj

REVIEW_TEXT_MODEL_NAMES = {"deepconn", "narre", "transnet", "daml"}


class SingleArgModelFactory(Protocol):
    def __call__(self, cfg: DictConfig) -> torch.nn.Module: ...


class AdjModelFactory(Protocol):
    def __call__(self, cfg: DictConfig, norm_adj: torch.Tensor) -> torch.nn.Module: ...


def _maybe_report_training_state(cfg: DictConfig) -> bool:
    model_name = str(cfg.model_name)
    missing: list[str] = []

    if model_name not in MODEL_DICT:
        missing.append("model registry")
    if model_name not in MODEL_TRAINER_DICT:
        missing.append("trainer registry")
    if model_name not in DATASET_DICT:
        missing.append("dataset registry")

    if missing:
        joined = ", ".join(missing)
        print(f"Skip trainer bootstrap for '{model_name}': missing {joined}.")
        return False
    return True


def _dataset_size(loader: DataLoader[object]) -> int:
    dataset = loader.dataset
    if isinstance(dataset, Sized):
        return len(dataset)
    return 0


def _load_model_checkpoint(model: torch.nn.Module, checkpoint_path: object, device: torch.device) -> dict[str, object]:
    if checkpoint_path in {None, "", "null", "None"}:
        raise ValueError("evaluation.checkpoint_path must be set for eval_only runs.")
    checkpoint_path = str(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_dict = dict(checkpoint)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    return checkpoint_dict


def _checkpoint_path_to_str(checkpoint_path: object) -> str:
    if checkpoint_path in {None, "", "null", "None"}:
        raise ValueError("evaluation.checkpoint_path must be set for eval_only runs.")
    return str(checkpoint_path)


def _maybe_set_rgcl_hidden_dim_from_checkpoint(cfg: DictConfig) -> None:
    checkpoint_path = _checkpoint_path_to_str(cfg.evaluation.checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = dict(checkpoint)["model_state_dict"]
    hidden_dim = int(state_dict["encoder.user_fc.weight"].shape[0])
    current_hidden_dim = int(cfg.model.hidden_dim)
    if hidden_dim != current_hidden_dim:
        print(
            "RGCL eval_only: overriding model.hidden_dim "
            f"from {current_hidden_dim} to {hidden_dim} to match checkpoint."
        )
        with open_dict(cfg):
            cfg.model.hidden_dim = hidden_dim


def _make_subset_loader(loader: DataLoader[object], ids: NDArray[np.int64]) -> DataLoader[object]:
    dataset = loader.dataset
    subset = Subset(dataset, ids.astype(np.int64).tolist())
    return DataLoader(subset, batch_size=loader.batch_size, shuffle=False)


def _evaluate_sentiment_subsets(cfg: DictConfig, trainer: object, test_loader: DataLoader[object]) -> dict[str, object]:
    subset_name = cfg.evaluation.get("sentiment_subset")
    if subset_name in {None, "", "null", "None"}:
        return {}
    subset_name = str(subset_name)
    if subset_name not in {"sentiment_pos", "sentiment_neg"}:
        raise ValueError("evaluation.sentiment_subset must be 'sentiment_pos' or 'sentiment_neg'")

    subset_dir = os.path.join(cfg.data.root, cfg.data.dataset, subset_name)
    results = {}
    for status in ["consistent", "inconsistent"]:
        id_path = os.path.join(subset_dir, f"{status}_id.npy")
        if not os.path.exists(id_path):
            raise FileNotFoundError(f"Missing sentiment subset IDs: {id_path}")
        ids = np.load(id_path).astype(np.int64)
        if len(ids) == 0:
            results[status] = {"num_samples": 0}
            continue
        metrics = getattr(trainer, "evaluate")(_make_subset_loader(test_loader, ids))
        results[status] = {"num_samples": int(len(ids)), **metrics}
    return results


def _evaluate_rgcl_sentiment_subsets(cfg: DictConfig, trainer: object, dataset: object) -> dict[str, object]:
    subset_name = cfg.evaluation.get("sentiment_subset")
    if subset_name in {None, "", "null", "None"}:
        return {}
    subset_name = str(subset_name)
    if subset_name not in {"sentiment_pos", "sentiment_neg"}:
        raise ValueError("evaluation.sentiment_subset must be 'sentiment_pos' or 'sentiment_neg'")

    subset_dir = os.path.join(cfg.data.root, cfg.data.dataset, subset_name)
    results = {}
    for status in ["consistent", "inconsistent"]:
        id_path = os.path.join(subset_dir, f"{status}_id.npy")
        if not os.path.exists(id_path):
            raise FileNotFoundError(f"Missing sentiment subset IDs: {id_path}")
        ids = np.load(id_path).astype(np.int64)
        if len(ids) == 0:
            results[status] = {"num_samples": 0}
            continue
        predictions, truths = getattr(dataset, "evaluate_test_subset")(ids, getattr(trainer, "model"))
        metrics = compute_all_metrics(predictions, truths)
        results[status] = {"num_samples": int(len(ids)), **metrics}
    return results


def _save_eval_only_results(cfg: DictConfig, results: dict[str, object]) -> None:
    os.makedirs(cfg.experiment.save_dir, exist_ok=True)
    output_path = os.path.join(cfg.experiment.save_dir, "eval_only_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Eval-only metrics saved to {output_path}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    debug_enabled = bool(cfg.experiment.get("debug", False))
    fast_dev_enabled = bool(cfg.experiment.get("fast_dev_run", False))
    if debug_enabled or fast_dev_enabled:
        with open_dict(cfg):
            cfg.experiment.fast_dev_run = True
            cfg.training.epoch = min(int(cfg.training.epoch), 1)
            cfg.training.batch = min(int(cfg.training.batch), 32)
            cfg.training.eval_batch = min(int(cfg.training.eval_batch), 64)
            cfg.evaluation.early_stop_patience = 1
    print(OmegaConf.to_yaml(cfg))

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{cfg.model_name}_{cfg.data.dataset}_{cfg.experiment.seed}_{current_time}"

    with open_dict(cfg):
        cfg.experiment.save_dir = os.path.join(cfg.experiment.save_dir, run_dir_name)

    os.makedirs(cfg.experiment.save_dir, exist_ok=True)
    print(f"Save directory: {cfg.experiment.save_dir}")

    set_seed(cfg.experiment.seed)

    device_str = f"cuda:{cfg.experiment.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    model_name = str(cfg.model_name).lower()

    if model_name not in MODEL_DICT:
        raise ValueError(f"Unknown model_name: {model_name}")
    if model_name not in MODEL_TRAINER_DICT:
        raise ValueError(f"No trainer registered for model_name: {model_name}")
    if model_name not in DATASET_DICT:
        raise ValueError(f"No dataset registered for model_name: {model_name}")

    cfg = set_stats_from_npy(cfg)

    train_loader, valid_loader, test_loader = get_dataloader(cfg, model_name)

    model_cls = MODEL_DICT[model_name]
    trainer_cls = MODEL_TRAINER_DICT[model_name]

    if model_name == "lightgcn":
        norm_adj = build_lightgcn_norm_adj_from_train(cfg).to(device)
        model = cast(AdjModelFactory, model_cls)(cfg, norm_adj).to(device)
    elif model_name in {"recafr", "mymodel_v3"}:
        norm_adj = build_recafr_norm_adj(cfg).to(device)
        model = cast(AdjModelFactory, model_cls)(cfg, norm_adj).to(device)
    else:
        model = cast(SingleArgModelFactory, model_cls)(cfg).to(device)
    trainer = trainer_cls(model, cfg, device)
    if bool(cfg.evaluation.get("eval_only", False)):
        if model_name == "rgcl":
            _maybe_set_rgcl_hidden_dim_from_checkpoint(cfg)
            getattr(trainer, "_configure")(train_loader)
        _load_model_checkpoint(model, cfg.evaluation.checkpoint_path, device)
        if model_name == "rgcl":
            test_metrics = getattr(trainer, "evaluate")(test_loader, "test")
            sentiment_results = _evaluate_rgcl_sentiment_subsets(cfg, trainer, test_loader)
        else:
            test_metrics = getattr(trainer, "evaluate")(test_loader)
            sentiment_results = _evaluate_sentiment_subsets(cfg, trainer, cast(DataLoader[object], test_loader))
        results: dict[str, object] = {"test_metrics": test_metrics}
        if sentiment_results:
            results["sentiment_subset"] = str(cfg.evaluation.sentiment_subset)
            results["sentiment_metrics"] = sentiment_results
        _save_eval_only_results(cfg, results)
        return

    getattr(trainer, "train")(train_loader, valid_loader, test_loader)
    

if __name__ == "__main__":
    main()
