import os
from typing import Union

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch.utils.data import Dataset


def _get_data_dir(cfg: DictConfig) -> str:
    data_type = str(cfg.data.get("type", "bert"))
    if data_type.lower() in {"none", "null", ""}:
        data_type = "bert"
    return os.path.join(str(cfg.data.root), str(cfg.data.dataset), data_type)


class MyModelV3Dataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, cfg: DictConfig, split: str = "train"):
        super().__init__()
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self.cfg = cfg
        self.split = split
        self.data_dir = _get_data_dir(cfg)

        user_ids = self._load_array(f"{split}_user_id.npy", np.int64)
        item_ids = self._load_array(f"{split}_item_id.npy", np.int64)
        ratings = self._load_array(f"{split}_rating.npy", np.float32)
        user_review_emb = self._load_array("user_review_emb.npy", np.float32)
        item_review_emb = self._load_array("item_review_emb.npy", np.float32)

        if not (len(user_ids) == len(item_ids) == len(ratings)):
            raise ValueError(
                f"Length mismatch in {split}: "
                f"users={len(user_ids)}, items={len(item_ids)}, ratings={len(ratings)}"
            )

        max_user_id = int(user_ids.max()) if len(user_ids) > 0 else -1
        max_item_id = int(item_ids.max()) if len(item_ids) > 0 else -1
        if max_user_id >= user_review_emb.shape[0]:
            raise ValueError(
                f"user_review_emb.npy has {user_review_emb.shape[0]} rows, "
                f"but {split} contains user_id={max_user_id}."
            )
        if max_item_id >= item_review_emb.shape[0]:
            raise ValueError(
                f"item_review_emb.npy has {item_review_emb.shape[0]} rows, "
                f"but {split} contains item_id={max_item_id}."
            )

        self.user_ids = torch.from_numpy(user_ids).long()
        self.item_ids = torch.from_numpy(item_ids).long()
        self.ratings = torch.from_numpy(ratings).float()
        self.user_review_emb = torch.from_numpy(user_review_emb).float()
        self.item_review_emb = torch.from_numpy(item_review_emb).float()

    def _load_array(
        self,
        filename: str,
        dtype: Union[type[np.int64], type[np.float32]],
    ) -> Union[NDArray[np.int64], NDArray[np.float32]]:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing MyModelV3 dataset file: {path}")
        return np.load(path).astype(dtype)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        return {
            "user_id": user_id,
            "item_id": item_id,
            "user_review": self.user_review_emb[user_id],
            "item_review": self.item_review_emb[item_id],
            "rating": self.ratings[idx],
        }
