import os

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torch.utils.data import Dataset


class DAMLDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.split = split

        user_ids, item_ids, ratings, user_docs, item_docs = self._load_data(split)

        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.user_docs = torch.tensor(user_docs, dtype=torch.long)
        self.item_docs = torch.tensor(item_docs, dtype=torch.long)

        assert len(self.user_ids) == len(self.item_ids) == len(self.ratings)
        assert len(self.user_ids) == len(self.user_docs) == len(self.item_docs)

        if self.user_docs.ndim != 2:
            raise ValueError(
                f"user_docs must be 2D [num_samples, doc_len], "
                f"got {tuple(self.user_docs.shape)}"
            )
        if self.item_docs.ndim != 2:
            raise ValueError(
                f"item_docs must be 2D [num_samples, doc_len], "
                f"got {tuple(self.item_docs.shape)}"
            )
        if self.user_docs.shape[1:] != self.item_docs.shape[1:]:
            raise ValueError(
                f"user_docs and item_docs shape mismatch: "
                f"user={tuple(self.user_docs.shape)}, item={tuple(self.item_docs.shape)}"
            )

        with open_dict(self.cfg):
            self.cfg.data.doc_len = int(self.user_docs.shape[1])
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
        }

    def _load_data(self, split: str):
        if split not in ["train", "valid", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'valid', 'test'].")

        data_dir = os.path.join(self.cfg.data.root, self.cfg.data.dataset, self.cfg.data.type)

        user_id_path = os.path.join(data_dir, f"{split}_user_id.npy")
        item_id_path = os.path.join(data_dir, f"{split}_item_id.npy")
        rating_path = os.path.join(data_dir, f"{split}_rating.npy")
        user_doc_path = os.path.join(data_dir, f"{split}_user_doc.npy")
        item_doc_path = os.path.join(data_dir, f"{split}_item_doc.npy")

        required_paths = [
            user_id_path,
            item_id_path,
            rating_path,
            user_doc_path,
            item_doc_path,
        ]
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing DAML dataset file: {path}")

        user_docs = np.load(user_doc_path).astype(np.int64)
        item_docs = np.load(item_doc_path).astype(np.int64)

        user_docs = self._to_daml_doc(user_docs)
        item_docs = self._to_daml_doc(item_docs)

        return (
            np.load(user_id_path),
            np.load(item_id_path),
            np.load(rating_path),
            user_docs,
            item_docs,
        )

    @staticmethod
    def _to_daml_doc(docs):
        if docs.ndim == 2:
            return docs
        if docs.ndim == 3:
            num_samples, review_count, review_length = docs.shape
            return docs.reshape(num_samples, review_count * review_length)
        raise ValueError(
            f"DAML token doc must be 2D [N, doc_len] or "
            f"3D [N, review_count, review_length], got {docs.shape}"
        )

    def _load_word_dim(self) -> int:
        data_dir = os.path.join(self.cfg.data.root, self.cfg.data.dataset, self.cfg.data.type)
        word_emb_path = os.path.join(data_dir, "word_emb.npy")
        if not os.path.exists(word_emb_path):
            raise FileNotFoundError(f"Missing word_emb.npy: {word_emb_path}. Run GloVe preprocessing again.")
        return int(np.load(word_emb_path, mmap_mode="r").shape[1])
