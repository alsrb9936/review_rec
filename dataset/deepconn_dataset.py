import torch
from torch.utils.data import Dataset
import pandas as pd
from omegaconf import DictConfig
from typing import Optional
from dataset.base_dataset import BaseDataset


class DeepCoNNDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: DictConfig,
        word_dict: dict,
        split: str = "train",
        history_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.word_dict = word_dict
        self.review_length = int(cfg.data.review_length)
        self.review_count = int(cfg.data.review_count)
        self.pad_id = int(cfg.data.pad_id)
        self.lowest_r_count = int(cfg.data.lowest_review_count)

        self.retain_rui = False
        if split == "train":
            self.retain_rui = bool(cfg.data.retain_rui)

        target_df = df.copy().reset_index(drop=True)

        if history_df is None:
            history_df = df
        history_df = history_df.copy().reset_index(drop=True)

        target_df["review_text"] = target_df["review_text"].apply(self._review2id)
        history_df["review_text"] = history_df["review_text"].apply(self._review2id)

        self.sparse_idx = set()

        user_reviews = self._get_reviews(
            target_df=target_df,
            history_df=history_df,
            lead="user_id",
            costar="item_id",
        )

        item_reviews = self._get_reviews(
            target_df=target_df,
            history_df=history_df,
            lead="item_id",
            costar="user_id",
        )

        user_ids = torch.tensor(target_df["user_id"].values, dtype=torch.long)
        item_ids = torch.tensor(target_df["item_id"].values, dtype=torch.long)
        ratings = torch.tensor(target_df["rating"].values, dtype=torch.float32).view(-1, 1)

        keep_idx = [idx for idx in range(len(target_df)) if idx not in self.sparse_idx]

        self.user_ids = user_ids[keep_idx]
        self.item_ids = item_ids[keep_idx]
        self.user_reviews = user_reviews[keep_idx]
        self.item_reviews = item_reviews[keep_idx]
        self.ratings = ratings[keep_idx]

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
            "user_reviews": self.user_reviews[idx],
            "item_reviews": self.item_reviews[idx],
        }

    def __len__(self):
        return self.ratings.shape[0]

    def _get_reviews(self, target_df, history_df, lead="user_id", costar="item_id"):
        reviews_by_lead = {
            lead_id: group[[costar, "review_text"]]
            for lead_id, group in history_df.groupby(lead)
        }

        lead_reviews = []

        for idx, (lead_id, costar_id) in enumerate(zip(target_df[lead], target_df[costar])):
            group = reviews_by_lead.get(lead_id)

            if group is None:
                reviews = []
            elif self.retain_rui:
                reviews = group["review_text"].to_list()
            else:
                reviews = group.loc[group[costar] != costar_id, "review_text"].to_list()

            if len(reviews) < self.lowest_r_count:
                self.sparse_idx.add(idx)

            reviews = self._adjust_review_list(
                reviews,
                self.review_length,
                self.review_count,
            )
            lead_reviews.append(reviews)

        return torch.LongTensor(lead_reviews)

    def _adjust_review_list(self, reviews, r_length, r_count):
        reviews = reviews[:r_count]

        pad_review = [self.pad_id] * r_length
        while len(reviews) < r_count:
            reviews.append(pad_review)

        fixed_reviews = []
        for review in reviews:
            review = review[:r_length]
            review = review + [self.pad_id] * (r_length - len(review))
            fixed_reviews.append(review)

        return fixed_reviews

    def _review2id(self, review):
        if not isinstance(review, str):
            return []

        wids = []
        for word in review.split():
            if word in self.word_dict:
                wids.append(self.word_dict[word])
            else:
                wids.append(self.pad_id)
        return wids