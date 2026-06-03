import os

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torch.utils.data import Dataset


class RMGDataset(Dataset):
    """Dataset for Review Meets Graph style rating prediction.

    Required GloVe files per split:
        {split}_user_id.npy, {split}_item_id.npy, {split}_rating.npy
        {split}_user_doc.npy, {split}_item_doc.npy
        {split}_user_review_item_ids.npy, {split}_item_review_user_ids.npy

    The first-order neighbor arrays are loaded directly. The second-order graph
    inputs are generated from the train split:
        user_item_user_ids: users connected to each item in user_review_item_ids
        item_user_item_ids: items connected to each user in item_review_user_ids
    """

    def __init__(self, cfg: DictConfig, split: str = "train"):
        super().__init__()
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self.cfg = cfg
        self.split = split
        self.data_dir = os.path.join(cfg.data.root, cfg.data.dataset, cfg.data.type)
        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)
        self.pad_user_id = self.num_users
        self.pad_item_id = self.num_items

        (
            user_ids,
            item_ids,
            ratings,
            user_docs,
            item_docs,
            user_review_item_ids,
            item_review_user_ids,
        ) = self._load_split(split)

        user_docs = self._to_hier_doc(user_docs, "user_doc")
        item_docs = self._to_hier_doc(item_docs, "item_doc")
        if user_docs.shape[1:] != item_docs.shape[1:]:
            raise ValueError(
                f"user_doc and item_doc shape mismatch: user={user_docs.shape}, item={item_docs.shape}"
            )
        if user_review_item_ids.shape != item_review_user_ids.shape:
            raise ValueError(
                f"neighbor shape mismatch: user_review_item_ids={user_review_item_ids.shape}, "
                f"item_review_user_ids={item_review_user_ids.shape}"
            )

        self.neighbor_count = int(user_review_item_ids.shape[1])
        user_to_items, item_to_users = self._build_train_neighbor_tables(self.neighbor_count)

        safe_user_review_item_ids = self._sanitize_ids(user_review_item_ids, self.num_items, self.pad_item_id)
        safe_item_review_user_ids = self._sanitize_ids(item_review_user_ids, self.num_users, self.pad_user_id)
        user_item_user_ids = item_to_users[safe_user_review_item_ids]
        item_user_item_ids = user_to_items[safe_item_review_user_ids]

        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.user_docs = torch.tensor(user_docs, dtype=torch.long)
        self.item_docs = torch.tensor(item_docs, dtype=torch.long)
        self.user_item_ids = torch.tensor(safe_user_review_item_ids, dtype=torch.long)
        self.item_user_ids = torch.tensor(safe_item_review_user_ids, dtype=torch.long)
        self.user_item_user_ids = torch.tensor(user_item_user_ids, dtype=torch.long)
        self.item_user_item_ids = torch.tensor(item_user_item_ids, dtype=torch.long)

        if not (
            len(self.user_ids)
            == len(self.item_ids)
            == len(self.ratings)
            == len(self.user_docs)
            == len(self.item_docs)
        ):
            raise ValueError("RMG split arrays must have the same first dimension.")

        with open_dict(self.cfg):
            self.cfg.data.review_count = int(self.user_docs.shape[1])
            self.cfg.data.sentence_count = int(self.user_docs.shape[2])
            self.cfg.data.sentence_length = int(self.user_docs.shape[3])
            self.cfg.data.neighbor_count = self.neighbor_count
            self.cfg.data.word_dim = self._load_word_dim()

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
            "user_doc": self.user_docs[idx],
            "item_doc": self.item_docs[idx],
            "user_item_ids": self.user_item_ids[idx],
            "item_user_ids": self.item_user_ids[idx],
            "user_item_user_ids": self.user_item_user_ids[idx],
            "item_user_item_ids": self.item_user_item_ids[idx],
        }

    def _load_split(self, split: str):
        paths = {
            "user_id": os.path.join(self.data_dir, f"{split}_user_id.npy"),
            "item_id": os.path.join(self.data_dir, f"{split}_item_id.npy"),
            "rating": os.path.join(self.data_dir, f"{split}_rating.npy"),
            "user_doc": os.path.join(self.data_dir, f"{split}_user_doc.npy"),
            "item_doc": os.path.join(self.data_dir, f"{split}_item_doc.npy"),
            "user_review_item_ids": os.path.join(self.data_dir, f"{split}_user_review_item_ids.npy"),
            "item_review_user_ids": os.path.join(self.data_dir, f"{split}_item_review_user_ids.npy"),
        }
        for path in paths.values():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing RMG dataset file: {path}")

        return (
            np.load(paths["user_id"]).astype(np.int64),
            np.load(paths["item_id"]).astype(np.int64),
            np.load(paths["rating"]).astype(np.float32),
            np.load(paths["user_doc"]).astype(np.int64),
            np.load(paths["item_doc"]).astype(np.int64),
            np.load(paths["user_review_item_ids"]).astype(np.int64),
            np.load(paths["item_review_user_ids"]).astype(np.int64),
        )

    @staticmethod
    def _to_hier_doc(docs: np.ndarray, name: str) -> np.ndarray:
        if docs.ndim == 4:
            return docs
        if docs.ndim == 3:
            return docs[:, :, None, :]
        raise ValueError(
            f"{name} must be 3D [N, review_count, review_length] or "
            f"4D [N, review_count, sentence_count, sentence_length], got {docs.shape}"
        )

    @staticmethod
    def _sanitize_ids(ids: np.ndarray, entity_count: int, pad_id: int) -> np.ndarray:
        ids = ids.copy()
        ids[(ids < 0) | (ids >= entity_count)] = pad_id
        return ids

    def _build_train_neighbor_tables(self, neighbor_count: int):
        train_user_id = np.load(os.path.join(self.data_dir, "train_user_id.npy")).astype(np.int64)
        train_item_id = np.load(os.path.join(self.data_dir, "train_item_id.npy")).astype(np.int64)
        train_user_review_item_ids = np.load(
            os.path.join(self.data_dir, "train_user_review_item_ids.npy")
        ).astype(np.int64)
        train_item_review_user_ids = np.load(
            os.path.join(self.data_dir, "train_item_review_user_ids.npy")
        ).astype(np.int64)

        user_to_items = np.full(
            (self.num_users + 1, neighbor_count),
            self.pad_item_id,
            dtype=np.int64,
        )
        item_to_users = np.full(
            (self.num_items + 1, neighbor_count),
            self.pad_user_id,
            dtype=np.int64,
        )

        seen_users = set()
        for user_id, item_neighbors in zip(train_user_id, train_user_review_item_ids):
            user_id = int(user_id)
            if user_id < 0 or user_id >= self.num_users or user_id in seen_users:
                continue
            user_to_items[user_id] = self._sanitize_ids(item_neighbors[:neighbor_count], self.num_items, self.pad_item_id)
            seen_users.add(user_id)

        seen_items = set()
        for item_id, user_neighbors in zip(train_item_id, train_item_review_user_ids):
            item_id = int(item_id)
            if item_id < 0 or item_id >= self.num_items or item_id in seen_items:
                continue
            item_to_users[item_id] = self._sanitize_ids(user_neighbors[:neighbor_count], self.num_users, self.pad_user_id)
            seen_items.add(item_id)

        return user_to_items, item_to_users

    def _load_word_dim(self) -> int:
        word_emb_path = os.path.join(self.data_dir, "word_emb.npy")
        if not os.path.exists(word_emb_path):
            raise FileNotFoundError(f"Missing word_emb.npy: {word_emb_path}. Run GloVe preprocessing again.")
        return int(np.load(word_emb_path, mmap_mode="r").shape[1])
