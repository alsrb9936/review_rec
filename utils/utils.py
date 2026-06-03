import os
import random

import numpy as np
import torch
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
    if cfg.model_name.lower() in GLOVE_MODEL_NAMES:
        cfg.data.type = "glove"
    elif cfg.model_name.lower() in BERT_MODEL_NAMES:
        cfg.data.type = "bert"

    data_dir = os.path.join(cfg.data.root, cfg.data.dataset, cfg.data.type)
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


def build_lightgcn_norm_adj_from_train(cfg: DictConfig) -> torch.Tensor:
    data_dir = os.path.join(cfg.data.root, cfg.data.dataset, cfg.data.type)
    user_path = os.path.join(data_dir, "train_user_id.npy")
    item_path = os.path.join(data_dir, "train_item_id.npy")
    if not os.path.exists(user_path):
        raise FileNotFoundError(f"Missing train user file: {user_path}")
    if not os.path.exists(item_path):
        raise FileNotFoundError(f"Missing train item file: {item_path}")

    user_ids = np.load(user_path).astype(np.int64)
    item_ids = np.load(item_path).astype(np.int64)
    num_users = int(cfg.stats.num_users)
    num_items = int(cfg.stats.num_items)
    num_nodes = num_users + num_items
    item_nodes = item_ids + num_users

    rows = np.concatenate([user_ids, item_nodes])
    cols = np.concatenate([item_nodes, user_ids])
    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes)).coalesce()

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    row, col = adj.indices()
    values = adj.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(adj.indices(), values, size=adj.shape).coalesce()


def get_dataloader(cfg, model_name):
    dataset_cls = DATASET_DICT[model_name]
    train_dataset = dataset_cls(cfg, split="train")
    valid_dataset = dataset_cls(cfg, split="valid")
    test_dataset = dataset_cls(cfg, split="test")

    if str(model_name).lower() == "rgcl":
        train_batch_size = len(train_dataset)
        train_shuffle = False
    else:
        train_batch_size = cfg.training.batch
        train_shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.training.eval_batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.eval_batch, shuffle=False)

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    if str(model_name).lower() == "rgcl":
        print(f"RGCL full-batch training enabled: batch_size={train_batch_size}")
    return train_loader, valid_loader, test_loader
