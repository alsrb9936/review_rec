import os
import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig


class LightGCNDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "train"):
        super().__init__()

        self.cfg = cfg
        self.split = split

        user_ids, item_ids, ratings = self._load_data(split)

        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

        assert len(self.user_ids) == len(self.item_ids) == len(self.ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
        }

    def _load_data(self, split: str):
        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Invalid split: {split}. Must be one of ['train', 'valid', 'test']."
            )

        data_dir = os.path.join(
            self.cfg.data.root,
            self.cfg.data.dataset,
            self.cfg.data.type,
        )

        user_path = os.path.join(data_dir, f"{split}_user_id.npy")
        item_path = os.path.join(data_dir, f"{split}_item_id.npy")
        rating_path = os.path.join(data_dir, f"{split}_rating.npy")

        user_ids = np.load(user_path)
        item_ids = np.load(item_path)
        ratings = np.load(rating_path)

        return user_ids, item_ids, ratings