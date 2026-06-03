# data/mymodel_dataset.py

import os

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


class MyModelDataset(Dataset):
    """
    Mini-batch rating dataset for MyModel.

    This dataset returns only user_id, item_id, rating.
    Review embeddings are loaded inside MyModel as train-only review graph buffers.
    """

    def __init__(self, cfg: DictConfig, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.split = split

        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split: {split}")

        data_type = str(cfg.data.get("type", "bert"))
        if data_type.lower() in {"none", "null", ""}:
            data_type = "bert"

        self.data_dir = os.path.join(cfg.data.root, cfg.data.dataset, data_type)

        user_path = os.path.join(self.data_dir, f"{split}_user_id.npy")
        item_path = os.path.join(self.data_dir, f"{split}_item_id.npy")
        rating_path = os.path.join(self.data_dir, f"{split}_rating.npy")

        for path in [user_path, item_path, rating_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset file: {path}")

        user_ids = np.load(user_path).astype(np.int64)
        item_ids = np.load(item_path).astype(np.int64)
        ratings = np.load(rating_path).astype(np.float32)

        if not (len(user_ids) == len(item_ids) == len(ratings)):
            raise ValueError(
                f"Length mismatch in {split}: "
                f"users={len(user_ids)}, items={len(item_ids)}, ratings={len(ratings)}"
            )

        self.user_ids = torch.from_numpy(user_ids).long()
        self.item_ids = torch.from_numpy(item_ids).long()
        self.ratings = torch.from_numpy(ratings).float()

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
        }