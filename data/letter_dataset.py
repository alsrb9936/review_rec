import os

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch.utils.data import Dataset


class LETTERDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, cfg: DictConfig, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.split = split

        user_ids, item_ids, ratings = self._load_data(split)
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

        if not (len(self.user_ids) == len(self.item_ids) == len(self.ratings)):
            raise ValueError(
                f"Length mismatch in {split}: "
                f"users={len(self.user_ids)}, items={len(self.item_ids)}, ratings={len(self.ratings)}"
            )

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
        }

    def _load_data(self, split: str) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]:
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'valid', 'test'].")

        data_dir = os.path.join(str(self.cfg.data.root), str(self.cfg.data.dataset), str(self.cfg.data.type))
        user_ids = np.load(os.path.join(data_dir, f"{split}_user_id.npy")).astype(np.int64)
        item_ids = np.load(os.path.join(data_dir, f"{split}_item_id.npy")).astype(np.int64)
        ratings = np.load(os.path.join(data_dir, f"{split}_rating.npy")).astype(np.float32)
        return user_ids, item_ids, ratings
