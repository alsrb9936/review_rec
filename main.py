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

    interactions = load_interaction_data(cfg)
    train_df, valid_df, test_df = split_by_ratio(
        interactions,
        train_ratio=cfg.data.split.train_ratio,
        valid_ratio=cfg.data.split.valid_ratio,
        random_state=cfg.experiment.seed,
    )

    print(
        f"Loaded interactions "
        f"(train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}) "
        f"with columns={list(interactions.columns)}"
    )

    train_loader, valid_loader, test_loader = get_dataloader(train_df, valid_df, test_df, cfg)
    print(f"Created dataloaders for model='{cfg.model_name}'")

    model_name = cfg.model_name
    if _maybe_report_training_state(cfg):
        model = MODEL_DICT[model_name](cfg).to(device)
        trainer = MODEL_TRAINER_DICT[model_name](model, cfg, device)
        trainer.train(train_loader, valid_loader, test_loader)


if __name__ == "__main__":
    main()
