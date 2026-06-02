# utils/utils.py
import torch
import random
import os
import numpy as np
from omegaconf import DictConfig, open_dict

from torch.utils.data import DataLoader
from data import DATASET_DICT
from models import MODEL_DICT
from trainer import MODEL_TRAINER_DICT

GLOVE_MODEL_NAMES = {"deepconn", "narre", "transnet", "daml", "neumf", "lightgcn"}
BERT_MODEL_NAMES = {"rgcl", "letter", "recafr"}
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def set_stats_from_npy(cfg: DictConfig) -> DictConfig:
    """
    Set cfg.stats.num_users and cfg.stats.num_items from saved npy files.

    Expected files:
        {cfg.data.root}/{cfg.data.dataset}/{cfg.data.type}/{split}_user_id.npy
        {cfg.data.root}/{cfg.data.dataset}/{cfg.data.type}/{split}_item_id.npy
    """

    if cfg.model_name.lower() in GLOVE_MODEL_NAMES:
        cfg.data.type = "glove"
    elif cfg.model_name.lower() in BERT_MODEL_NAMES:
        cfg.data.type = "bert"

    data_dir = os.path.join(
        cfg.data.root,
        cfg.data.dataset,
        cfg.data.type
    )

    max_user_id = -1
    max_item_id = -1

    for split in ["train", "valid", "test"]:
        user_path = os.path.join(data_dir, f"{split}_user_id.npy")
        item_path = os.path.join(data_dir, f"{split}_item_id.npy")

        if not os.path.exists(user_path):
            raise FileNotFoundError(f"Missing user id file: {user_path}")
        if not os.path.exists(item_path):
            raise FileNotFoundError(f"Missing item id file: {item_path}")

        user_ids = np.load(user_path)
        item_ids = np.load(item_path)

        if len(user_ids) == 0:
            raise ValueError(f"Empty user id file: {user_path}")
        if len(item_ids) == 0:
            raise ValueError(f"Empty item id file: {item_path}")

        max_user_id = max(max_user_id, int(user_ids.max()))
        max_item_id = max(max_item_id, int(item_ids.max()))

    with open_dict(cfg):
        cfg.stats.num_users = max_user_id + 1
        cfg.stats.num_items = max_item_id + 1

    print(f"num_users: {cfg.stats.num_users}")
    print(f"num_items: {cfg.stats.num_items}")

    return cfg

def get_dataloader(cfg, model_name):
    dataset_cls = DATASET_DICT[model_name]
    train_dataset = dataset_cls(cfg, split="train")
    valid_dataset = dataset_cls(cfg, split="valid")
    test_dataset = dataset_cls(cfg, split="test")

    train_loader = DataLoader(train_dataset,batch_size=cfg.training.batch,shuffle=True)
    valid_loader = DataLoader(valid_dataset,batch_size=cfg.training.eval_batch,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=cfg.training.eval_batch,shuffle=False)
    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    return train_loader, valid_loader, test_loader
