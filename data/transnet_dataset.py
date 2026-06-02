import os

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torch.utils.data import Dataset


class TransNetDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.split = split

        user_ids, item_ids, ratings, user_docs, item_docs, target_docs = self._load_data(split)

        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.user_docs = torch.tensor(user_docs, dtype=torch.float32)
        self.item_docs = torch.tensor(item_docs, dtype=torch.float32)
        self.target_docs = None if target_docs is None else torch.tensor(target_docs, dtype=torch.float32)

        assert len(self.user_ids) == len(self.item_ids) == len(self.ratings)
        assert len(self.user_ids) == len(self.user_docs) == len(self.item_docs)
        if self.target_docs is not None:
            assert len(self.user_ids) == len(self.target_docs)

        if self.user_docs.ndim != 3:
            raise ValueError(
                f"user_docs must be 3D [num_samples, doc_len, word_dim], got {tuple(self.user_docs.shape)}"
            )
        if self.item_docs.ndim != 3:
            raise ValueError(
                f"item_docs must be 3D [num_samples, doc_len, word_dim], got {tuple(self.item_docs.shape)}"
            )
        if self.user_docs.shape[1:] != self.item_docs.shape[1:]:
            raise ValueError(
                f"user_docs and item_docs shape mismatch: "
                f"user={tuple(self.user_docs.shape)}, item={tuple(self.item_docs.shape)}"
            )
        if self.target_docs is not None and self.target_docs.ndim != 3:
            raise ValueError(
                f"target_docs must be 3D [num_samples, target_doc_len, word_dim], got {tuple(self.target_docs.shape)}"
            )
        if self.target_docs is not None and self.target_docs.shape[2] != self.user_docs.shape[2]:
            raise ValueError(
                f"target_docs word_dim mismatch: target={self.target_docs.shape[2]}, doc={self.user_docs.shape[2]}"
            )

        with open_dict(self.cfg):
            self.cfg.data.doc_len = int(self.user_docs.shape[1])
            self.cfg.data.word_dim = int(self.user_docs.shape[2])
            if self.target_docs is not None:
                self.cfg.data.target_doc_len = int(self.target_docs.shape[1])

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        sample = {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
            "user_doc": self.user_docs[idx],
            "item_doc": self.item_docs[idx],
        }
        if self.target_docs is not None:
            sample["target_doc"] = self.target_docs[idx]
        return sample

    def _load_data(self, split: str):
        if split not in ["train", "valid", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'valid', 'test'].")

        data_dir = os.path.join(self.cfg.data.root, self.cfg.data.dataset, self.cfg.data.type)

        user_id_path = os.path.join(data_dir, f"{split}_user_id.npy")
        item_id_path = os.path.join(data_dir, f"{split}_item_id.npy")
        rating_path = os.path.join(data_dir, f"{split}_rating.npy")
        user_doc_path = os.path.join(data_dir, f"{split}_user_doc_emb.npy")
        item_doc_path = os.path.join(data_dir, f"{split}_item_doc_emb.npy")

        required_paths = [user_id_path, item_id_path, rating_path, user_doc_path, item_doc_path]
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing TransNet dataset file: {path}")

        user_docs = self._to_sequence_doc(np.load(user_doc_path).astype(np.float32))
        item_docs = self._to_sequence_doc(np.load(item_doc_path).astype(np.float32))
        target_docs = None

        if split == "train":
            target_doc_emb_path = os.path.join(data_dir, "train_target_doc_emb.npy")
            if not os.path.exists(target_doc_emb_path):
                raise FileNotFoundError(
                    "TransNet training requires train_target_doc_emb.npy. "
                    f"Expected: {target_doc_emb_path}"
                )
            target_docs = self._to_sequence_doc(np.load(target_doc_emb_path).astype(np.float32))

        return (
            np.load(user_id_path),
            np.load(item_id_path),
            np.load(rating_path),
            user_docs,
            item_docs,
            target_docs,
        )

    @staticmethod
    def _to_sequence_doc(docs: np.ndarray) -> np.ndarray:
        if docs.ndim == 3:
            return docs
        if docs.ndim == 4:
            num_samples, review_count, review_length, word_dim = docs.shape
            return docs.reshape(num_samples, review_count * review_length, word_dim)
        raise ValueError(
            f"TransNet doc embedding must be 3D [N, doc_len, word_dim] or "
            f"4D [N, review_count, review_length, word_dim], got {docs.shape}"
        )
