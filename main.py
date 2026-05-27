import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from dataset import DATASET_DICT
from models import MODEL_DICT
from trainer import MODEL_TRAINER_DICT
from utils.util import load_interaction_data, set_seed, split_by_ratio, get_dataloader


def _maybe_report_training_state(cfg: DictConfig) -> None:
    model_name = cfg.model_name
    missing = []

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


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.experiment.seed)

    device_str = f"cuda:{cfg.experiment.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    train_loader, valid_loader, test_loader = get_dataloader(cfg)

    print(
        f"Loaded interactions "
        f"(train={len(train_loader.dataset)}, valid={len(valid_loader.dataset)}, test={len(test_loader.dataset)}) "
        f"with columns={list(load_interaction_data(cfg).columns)}"
    )

    print(f"Created dataloaders for model='{cfg.model_name}'")

    model_name = cfg.model_name
    if _maybe_report_training_state(cfg):
        model = MODEL_DICT[model_name](cfg).to(device)
        trainer = MODEL_TRAINER_DICT[model_name](model, cfg, device)
        trainer.train(train_loader, valid_loader, test_loader)


if __name__ == "__main__":
    main()
