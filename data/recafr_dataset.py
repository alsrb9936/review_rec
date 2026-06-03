import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _get_data_dir(cfg):
    data_type = str(cfg.data.get("type", "bert"))
    if data_type.lower() in {"none", "null", ""}:
        data_type = "bert"
    return os.path.join(cfg.data.root, cfg.data.dataset, data_type)


class RecAFRDataset(Dataset):
    """Rating-prediction dataset for RecAFR.

    Each sample contains ``user_id``, ``item_id`` and its explicit ``rating``.
    No negative sampling or full-ranking evaluation state is kept here.
    """

    def __init__(self, cfg, split: str = "train"):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.cfg = cfg
        self.split = split
        self.data_dir = _get_data_dir(cfg)

        user_ids, item_ids, ratings = self._load_split(split)
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return int(len(self.ratings))

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
        }

    def _load_split(self, split: str):
        user_path = os.path.join(self.data_dir, f"{split}_user_id.npy")
        item_path = os.path.join(self.data_dir, f"{split}_item_id.npy")
        rating_path = os.path.join(self.data_dir, f"{split}_rating.npy")

        for path in [user_path, item_path, rating_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing RecAFR dataset file: {path}")

        user_ids = np.load(user_path).astype(np.int64)
        item_ids = np.load(item_path).astype(np.int64)
        ratings = np.load(rating_path).astype(np.float32)

        if not (len(user_ids) == len(item_ids) == len(ratings)):
            raise ValueError(
                f"Length mismatch in {split}: "
                f"users={len(user_ids)}, items={len(item_ids)}, ratings={len(ratings)}"
            )
        return user_ids, item_ids, ratings
