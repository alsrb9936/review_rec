import os

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torch.utils.data import Dataset


class DeepCoNNDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.split = split

        user_ids, item_ids, ratings, user_reviews, item_reviews = self._load_data(split)

        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

        # Precomputed GloVe document embeddings.
        # Shape: [num_samples, review_count, review_length, word_dim]
        self.user_reviews = torch.tensor(user_reviews, dtype=torch.float32)
        self.item_reviews = torch.tensor(item_reviews, dtype=torch.float32)

        assert len(self.user_ids) == len(self.item_ids) == len(self.ratings)
        assert len(self.user_ids) == len(self.user_reviews) == len(self.item_reviews)

        if self.user_reviews.ndim != 4:
            raise ValueError(
                "Expected user_reviews to be 4D "
                "[num_samples, review_count, review_length, word_dim], "
                f"but got shape={tuple(self.user_reviews.shape)}"
            )
        if self.item_reviews.ndim != 4:
            raise ValueError(
                "Expected item_reviews to be 4D "
                "[num_samples, review_count, review_length, word_dim], "
                f"but got shape={tuple(self.item_reviews.shape)}"
            )

        if self.user_reviews.shape[1:] != self.item_reviews.shape[1:]:
            raise ValueError(
                "user_reviews and item_reviews must have the same "
                "[review_count, review_length, word_dim], "
                f"but got user={tuple(self.user_reviews.shape[1:])}, "
                f"item={tuple(self.item_reviews.shape[1:])}"
            )

        review_count = int(self.user_reviews.shape[1])
        review_length = int(self.user_reviews.shape[2])
        word_dim = int(self.user_reviews.shape[3])

        with open_dict(self.cfg):
            self.cfg.data.review_count = review_count
            self.cfg.data.review_length = review_length
            self.cfg.data.word_dim = word_dim

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
            "user_reviews": self.user_reviews[idx],
            "item_reviews": self.item_reviews[idx],
        }

    def __len__(self):
        return len(self.ratings)

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

        user_id_path = os.path.join(data_dir, f"{split}_user_id.npy")
        item_id_path = os.path.join(data_dir, f"{split}_item_id.npy")
        rating_path = os.path.join(data_dir, f"{split}_rating.npy")
        user_doc_path = os.path.join(data_dir, f"{split}_user_doc_emb.npy")
        item_doc_path = os.path.join(data_dir, f"{split}_item_doc_emb.npy")

        required_paths = [
            user_id_path,
            item_id_path,
            rating_path,
            user_doc_path,
            item_doc_path,
        ]
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing DeepCoNN dataset file: {path}")

        user_ids = np.load(user_id_path)
        item_ids = np.load(item_id_path)
        ratings = np.load(rating_path)
        user_reviews = np.load(user_doc_path).astype(np.float32)
        item_reviews = np.load(item_doc_path).astype(np.float32)

        return user_ids, item_ids, ratings, user_reviews, item_reviews
