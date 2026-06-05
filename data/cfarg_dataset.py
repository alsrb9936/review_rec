import os

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from typing import Any


def _get_data_dir(cfg: DictConfig) -> str:
    data_type = str(cfg.data.get("type", "bert"))
    if data_type.lower() in {"none", "null", ""}:
        data_type = "bert"
    return os.path.join(str(cfg.data.root), str(cfg.data.dataset), data_type)


class CFARGDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for CF-only, whole-review fusion, and CFARG experiments.

    BERT preprocessing stores user/item review profiles built from train reviews.
    For validation/test, this is leakage-safe because the target split reviews are
    not embedded. For train, ``exclude_target_review`` optionally builds a
    leave-one-out profile for the current interaction so the target review is not
    passed as input.
    """

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
        if not (len(user_ids) == len(item_ids) == len(ratings)):
            raise ValueError(
                f"Length mismatch in {split}: users={len(user_ids)}, "
                f"items={len(item_ids)}, ratings={len(ratings)}"
            )

        user_review_emb = self._load_array("user_review_emb.npy", np.float32)
        item_review_emb = self._load_array("item_review_emb.npy", np.float32)

        sample_user_review = user_review_emb[user_ids]
        sample_item_review = item_review_emb[item_ids]

        if split == "train" and bool(cfg.data.get("exclude_target_review", True)):
            sample_user_review, sample_item_review = self._build_train_leave_one_out_profiles(
                user_ids=user_ids,
                item_ids=item_ids,
            )

        sample_user_review = self._apply_noise(sample_user_review, user_review_emb, side="user")
        sample_item_review = self._apply_noise(sample_item_review, item_review_emb, side="item")

        self.user_ids = torch.from_numpy(user_ids).long()
        self.item_ids = torch.from_numpy(item_ids).long()
        self.ratings = torch.from_numpy(ratings).float()
        self.user_review = torch.from_numpy(sample_user_review.astype(np.float32)).float()
        self.item_review = torch.from_numpy(sample_item_review.astype(np.float32)).float()

    def _load_array(self, filename: str, dtype: object) -> Any:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing CFARG dataset file: {path}")
        return np.load(path).astype(dtype)

    def _build_train_leave_one_out_profiles(
        self,
        user_ids: Any,
        item_ids: Any,
    ) -> tuple[Any, Any]:
        review_emb = self._load_array("review_emb.npy", np.float32)
        train_user_ids = self._load_array("train_user_id.npy", np.int64)
        train_item_ids = self._load_array("train_item_id.npy", np.int64)
        if len(review_emb) != len(train_user_ids) or len(review_emb) != len(train_item_ids):
            raise ValueError(
                "review_emb.npy must align with train_user_id.npy and train_item_id.npy "
                f"for exclude_target_review=true: reviews={len(review_emb)}, "
                f"users={len(train_user_ids)}, items={len(train_item_ids)}"
            )

        num_users = int(self.cfg.stats.num_users)
        num_items = int(self.cfg.stats.num_items)
        dim = int(review_emb.shape[1])

        user_sum = np.zeros((num_users, dim), dtype=np.float32)
        item_sum = np.zeros((num_items, dim), dtype=np.float32)
        user_count = np.bincount(train_user_ids, minlength=num_users).astype(np.float32)
        item_count = np.bincount(train_item_ids, minlength=num_items).astype(np.float32)
        np.add.at(user_sum, train_user_ids, review_emb)
        np.add.at(item_sum, train_item_ids, review_emb)

        user_left_sum = user_sum[user_ids] - review_emb
        item_left_sum = item_sum[item_ids] - review_emb
        user_left_count = user_count[user_ids] - 1.0
        item_left_count = item_count[item_ids] - 1.0

        user_profiles = np.zeros_like(user_left_sum, dtype=np.float32)
        item_profiles = np.zeros_like(item_left_sum, dtype=np.float32)
        user_nonzero = user_left_count > 0
        item_nonzero = item_left_count > 0
        user_profiles[user_nonzero] = user_left_sum[user_nonzero] / user_left_count[user_nonzero, None]
        item_profiles[item_nonzero] = item_left_sum[item_nonzero] / item_left_count[item_nonzero, None]
        return user_profiles, item_profiles

    def _apply_noise(
        self,
        sample_emb: Any,
        entity_emb: Any,
        side: str,
    ) -> Any:
        noise_cfg = self.cfg.get("noise", {})
        if not bool(noise_cfg.get("enabled", False)):
            return sample_emb
        if str(noise_cfg.get("type", "random_review_replacement")) != "random_review_replacement":
            raise ValueError(f"Unsupported noise.type: {noise_cfg.get('type')}")

        noise_side = str(noise_cfg.get("side", "both"))
        if noise_side not in {"both", side}:
            return sample_emb

        ratio = float(noise_cfg.get("ratio", 0.0))
        if ratio <= 0.0:
            return sample_emb
        if ratio > 1.0:
            raise ValueError(f"noise.ratio must be in [0, 1], got {ratio}")

        seed_offset = {"train": 11, "valid": 17, "test": 23}[self.split]
        side_offset = 101 if side == "user" else 211
        rng = np.random.default_rng(int(self.cfg.experiment.seed) + seed_offset + side_offset)
        noisy = sample_emb.copy()
        mask = rng.random(len(noisy)) < ratio
        if mask.any():
            replacement_ids = rng.integers(0, len(entity_emb), size=int(mask.sum()))
            noisy[mask] = entity_emb[replacement_ids]
        return noisy

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "user_review": self.user_review[idx],
            "item_review": self.item_review[idx],
            "rating": self.ratings[idx],
        }
