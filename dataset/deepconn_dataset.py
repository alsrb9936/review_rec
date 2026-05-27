import torch
import pandas as pd
from omegaconf import DictConfig
from typing import Optional
from dataset.base_dataset import BaseDataset


class DeepCoNNDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: DictConfig,
        split: str = "train",
        user_review_bank: Optional[dict] = None,
        item_review_bank: Optional[dict] = None,
    ):
        super().__init__(df, cfg, split)

        self.review_length = int(cfg.data.review_length)
        self.review_count = int(cfg.data.review_count)
        self.pad_id = int(cfg.data.pad_id)

        self.user_review_bank = user_review_bank or {}
        self.item_review_bank = item_review_bank or {}

        self._build_review_tensors()

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx].view(1),
            "user_reviews": self.user_reviews[idx],
            "item_reviews": self.item_reviews[idx],
        }

    def _build_review_tensors(self):
        user_review_tensors = []
        item_review_tensors = []

        for row in self.df.itertuples(index=False):
            user_id = int(row.user_id)
            item_id = int(row.item_id)

            user_reviews = list(self.user_review_bank.get(user_id, []))
            item_reviews = list(self.item_review_bank.get(item_id, []))

            user_review_tensors.append(self._adjust_review_list(user_reviews))
            item_review_tensors.append(self._adjust_review_list(item_reviews))

        self.user_reviews = torch.tensor(user_review_tensors, dtype=torch.long)
        self.item_reviews = torch.tensor(item_review_tensors, dtype=torch.long)

    def _adjust_review_list(self, reviews):
        reviews = reviews[:self.review_count]

        adjusted = []
        for review in reviews:
            review = list(review)
            review = review[:self.review_length]
            review = review + [self.pad_id] * (self.review_length - len(review))
            adjusted.append(review)

        while len(adjusted) < self.review_count:
            adjusted.append([self.pad_id] * self.review_length)

        return adjusted
