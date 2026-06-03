# data/rgcl_dataset.py
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _get_data_dir(cfg):
    data_type = str(cfg.data.get("type", "bert"))
    if data_type.lower() in {"none", "null", ""}:
        data_type = "bert"

    return os.path.join(
        cfg.data.root,
        cfg.data.dataset,
        data_type,
    )


class RGCLDataset(Dataset):
    """Numpy-backed dataset for RGCL.

    The BERT preprocessing step saves review_emb.npy only for train rows.
    Therefore train returns review_feat for ED-MI, while valid/test return
    only user_id, item_id, and rating.
    """

    def __init__(self, cfg, split: str = "train"):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.cfg = cfg
        self.split = split
        self.data_dir = _get_data_dir(cfg)

        self.user_ids = self._load_long(f"{split}_user_id.npy")
        self.item_ids = self._load_long(f"{split}_item_id.npy")
        self.ratings = self._load_float(f"{split}_rating.npy")

        if not (len(self.user_ids) == len(self.item_ids) == len(self.ratings)):
            raise ValueError(
                f"Length mismatch in {split}: "
                f"users={len(self.user_ids)}, items={len(self.item_ids)}, ratings={len(self.ratings)}"
            )

        if split == "train":
            review_path = os.path.join(self.data_dir, "review_emb.npy")
            if not os.path.exists(review_path):
                raise FileNotFoundError(f"Missing train review embedding file: {review_path}")

            review_feat = np.load(review_path).astype(np.float32)
            if len(review_feat) != len(self.user_ids):
                raise ValueError(
                    f"review_emb.npy must align with train interactions: "
                    f"review_emb={len(review_feat)}, train={len(self.user_ids)}"
                )
            self.review_feat = torch.from_numpy(review_feat).float()
        else:
            self.review_feat = None

    def _load_long(self, filename: str) -> torch.Tensor:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        return torch.from_numpy(np.load(path).astype(np.int64)).long()

    def _load_float(self, filename: str) -> torch.Tensor:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        return torch.from_numpy(np.load(path).astype(np.float32)).float()

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        out = {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
        }

        if self.split == "train":
            out["review_feat"] = self.review_feat[idx]

        return out
