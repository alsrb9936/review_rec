import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


def _get_data_dir(cfg):
    data_type = str(cfg.data.get("type", "bert"))
    if data_type.lower() in {"none", "null", ""}:
        data_type = "bert"
    return os.path.join(cfg.data.root, cfg.data.dataset, data_type)


class RecAFRDataset(Dataset):
    """Pairwise implicit-feedback dataset for RecAFR.

    Train samples are ``(user, positive item, sampled negative item)`` triples.
    Validation/test samples iterate over users and keep the positive-item sets on
    the dataset object so the trainer can compute full-ranking Recall/NDCG.
    """

    def __init__(self, cfg, split: str = "train"):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.cfg = cfg
        self.split = split
        self.data_dir = _get_data_dir(cfg)
        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)

        self.user_ids, self.item_ids, self.ratings = self._load_split(split)
        self.train_user_ids, self.train_item_ids, self.train_ratings = self._load_split("train")

        self.train_user_pos = self._build_user_pos(self.train_user_ids, self.train_item_ids)
        self.eval_user_pos = self._build_user_pos(self.user_ids, self.item_ids)
        self.eval_users = np.array(sorted(self.eval_user_pos.keys()), dtype=np.int64)

    def __len__(self):
        if self.split == "train":
            return int(len(self.user_ids))
        return int(len(self.eval_users))

    def __getitem__(self, idx):
        if self.split == "train":
            user_id = int(self.user_ids[idx])
            pos_item_id = int(self.item_ids[idx])
            neg_item_id = self._sample_negative(user_id)
            return {
                "user_id": torch.tensor(user_id, dtype=torch.long),
                "pos_item_id": torch.tensor(pos_item_id, dtype=torch.long),
                "neg_item_id": torch.tensor(neg_item_id, dtype=torch.long),
            }

        user_id = int(self.eval_users[idx])
        return {"user_id": torch.tensor(user_id, dtype=torch.long)}

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

    @staticmethod
    def _build_user_pos(user_ids, item_ids):
        user_pos = defaultdict(set)
        for user_id, item_id in zip(user_ids, item_ids):
            user_pos[int(user_id)].add(int(item_id))
        return dict(user_pos)

    def _sample_negative(self, user_id: int) -> int:
        positives = self.train_user_pos.get(int(user_id), set())
        if len(positives) >= self.num_items:
            raise ValueError(f"User {user_id} has no available negative items.")

        neg_item_id = int(np.random.randint(0, self.num_items))
        while neg_item_id in positives:
            neg_item_id = int(np.random.randint(0, self.num_items))
        return neg_item_id
