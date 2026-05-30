import os
from collections.abc import Sized
from datetime import datetime

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from dataset import DATASET_DICT
from models import MODEL_DICT
from trainer import MODEL_TRAINER_DICT
from utils.util import load_interaction_data, set_seed, get_dataloader

REVIEW_TEXT_MODEL_NAMES = {"deepconn", "narre", "transnet"}
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


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
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
    model_name = str(cfg.model_name)
    model_name = cfg.model_name.lower()
    if not _maybe_report_training_state(cfg):
        return

    if model_name in REVIEW_TEXT_MODEL_NAMES:
        train_loader, valid_loader, test_loader, word_emb, _ = get_dataloader(cfg)
        model = MODEL_DICT[model_name](cfg,word_emb).to(device)

    elif model_name in [
        "mymodel",
        "rgcl"
    ]:
        train_loader, valid_loader, test_loader, graph_obj = get_dataloader(cfg)
        model = MODEL_DICT[model_name](cfg, graph_obj).to(device)
    else:
        train_loader, valid_loader, test_loader = get_dataloader(cfg)
        model = MODEL_DICT[model_name](cfg).to(device)
    print(
        f"Loaded interactions "
        f"(train={_dataset_size(train_loader)}, valid={_dataset_size(valid_loader)}, test={_dataset_size(test_loader)}) "
    )

    print(f"Created dataloaders for model='{cfg.model_name}'")

    
    trainer = MODEL_TRAINER_DICT[model_name](model, cfg, device)
    trainer.train(train_loader, valid_loader, test_loader)


if __name__ == "__main__":
    main()
