import os

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torch.utils.data import Dataset


class NARREDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.split = split

        (
            user_ids,
            item_ids,
            ratings,
            user_reviews,
            item_reviews,
            user_review_item_ids,
            item_review_user_ids,
        ) = self._load_data(split)

        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.user_reviews = torch.tensor(user_reviews, dtype=torch.float32)
        self.item_reviews = torch.tensor(item_reviews, dtype=torch.float32)
        self.user_review_item_ids = torch.tensor(user_review_item_ids, dtype=torch.long)
        self.item_review_user_ids = torch.tensor(item_review_user_ids, dtype=torch.long)

        assert len(self.user_ids) == len(self.item_ids) == len(self.ratings)
        assert len(self.user_ids) == len(self.user_reviews) == len(self.item_reviews)
        assert len(self.user_ids) == len(self.user_review_item_ids)
        assert len(self.user_ids) == len(self.item_review_user_ids)

        if self.user_reviews.ndim != 4:
            raise ValueError(f"user_reviews must be 4D [N, review_count, review_length, word_dim], got {tuple(self.user_reviews.shape)}")
        if self.item_reviews.ndim != 4:
            raise ValueError(f"item_reviews must be 4D [N, review_count, review_length, word_dim], got {tuple(self.item_reviews.shape)}")
        if self.user_reviews.shape[1:] != self.item_reviews.shape[1:]:
            raise ValueError("user_reviews and item_reviews shape mismatch")
        if self.user_review_item_ids.shape != self.user_reviews.shape[:2]:
            raise ValueError("user_review_item_ids must be [N, review_count]")
        if self.item_review_user_ids.shape != self.item_reviews.shape[:2]:
            raise ValueError("item_review_user_ids must be [N, review_count]")

        with open_dict(self.cfg):
            self.cfg.data.review_count = int(self.user_reviews.shape[1])
            self.cfg.data.review_length = int(self.user_reviews.shape[2])
            self.cfg.data.word_dim = int(self.user_reviews.shape[3])

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
            "user_reviews": self.user_reviews[idx],
            "user_review_item_ids": self.user_review_item_ids[idx],
            "item_reviews": self.item_reviews[idx],
            "item_review_user_ids": self.item_review_user_ids[idx],
        }

    def _load_data(self, split: str):
        if split not in ["train", "valid", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'valid', 'test'].")

        data_dir = os.path.join(self.cfg.data.root, self.cfg.data.dataset, self.cfg.data.type)

        user_id_path = os.path.join(data_dir, f"{split}_user_id.npy")
        item_id_path = os.path.join(data_dir, f"{split}_item_id.npy")
        rating_path = os.path.join(data_dir, f"{split}_rating.npy")
        user_doc_path = os.path.join(data_dir, f"{split}_user_doc_emb.npy")
        item_doc_path = os.path.join(data_dir, f"{split}_item_doc_emb.npy")
        user_review_item_ids_path = os.path.join(data_dir, f"{split}_user_review_item_ids.npy")
        item_review_user_ids_path = os.path.join(data_dir, f"{split}_item_review_user_ids.npy")

        required_paths = [
            user_id_path,
            item_id_path,
            rating_path,
            user_doc_path,
            item_doc_path,
            user_review_item_ids_path,
            item_review_user_ids_path,
        ]
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing NARRE dataset file: {path}")

        return (
            np.load(user_id_path),
            np.load(item_id_path),
            np.load(rating_path),
            np.load(user_doc_path).astype(np.float32),
            np.load(item_doc_path).astype(np.float32),
            np.load(user_review_item_ids_path),
            np.load(item_review_user_ids_path),
        )
