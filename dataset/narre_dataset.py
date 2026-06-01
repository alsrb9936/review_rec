from dataclasses import dataclass
from typing import Dict, List

import torch
from omegaconf import DictConfig

from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from gensim.models.keyedvectors import Word2VecKeyedVectors

import torch
from torch.utils.data import Dataset
import pandas as pd


class NARREDataset(Dataset):
    def __init__(self, df, cfg, word_dict, split="train", history_df=None):
        super().__init__()

        self.df = df.copy().reset_index(drop=True)

        if history_df is None:
            history_df = df
        self.history_df = history_df.copy().reset_index(drop=True)

        self.word_dict = word_dict

        self.review_length = int(cfg.data.review_length)
        self.review_count = int(cfg.data.review_count)
        self.pad_id = int(cfg.data.pad_id)

        self.pad_user_id = int(cfg.stats.num_users)
        self.pad_item_id = int(cfg.stats.num_items)
        self.lowest_r_count = int(cfg.data.lowest_review_count)

        self.retain_rui = False
        if split == "train":
            self.retain_rui = bool(cfg.data.retain_rui)

        self.df["review_text"] = self.df["review_text"].apply(self._review2id)
        self.history_df["review_text"] = self.history_df["review_text"].apply(self._review2id)

        self.user_reviews, self.user_review_item_ids = self._get_reviews_and_ids(
            target_df=self.df,
            history_df=self.history_df,
            lead="user_id",
            costar="item_id",
        )

        self.item_reviews, self.item_review_user_ids = self._get_reviews_and_ids(
            target_df=self.df,
            history_df=self.history_df,
            lead="item_id",
            costar="user_id",
        )

        self.user_ids = torch.tensor(self.df["user_id"].values, dtype=torch.long)
        self.item_ids = torch.tensor(self.df["item_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(self.df["rating"].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.ratings.shape[0]

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

    def _get_reviews_and_ids(self, target_df, history_df, lead, costar):
        reviews_by_lead = {
            lead_id: group[[costar, "review_text"]]
            for lead_id, group in history_df.groupby(lead)
        }

        if costar == "item_id":
            pad_costar_id = self.pad_item_id
        elif costar == "user_id":
            pad_costar_id = self.pad_user_id
        else:
            raise ValueError(f"Unknown costar column: {costar}")

        all_reviews = []
        all_costar_ids = []

        for idx, (lead_id, costar_id) in enumerate(zip(target_df[lead], target_df[costar])):
            group = reviews_by_lead.get(lead_id)

            if group is None:
                reviews = []
                costar_ids = []
            elif self.retain_rui:
                reviews = group["review_text"].to_list()
                costar_ids = group[costar].to_list()
            else:
                filtered = group[group[costar] != costar_id]
                reviews = filtered["review_text"].to_list()
                costar_ids = filtered[costar].to_list()


            reviews, costar_ids = self._adjust_review_list_and_ids(
                reviews,
                costar_ids,
                pad_costar_id,
            )

            all_reviews.append(reviews)
            all_costar_ids.append(costar_ids)

        return torch.LongTensor(all_reviews), torch.LongTensor(all_costar_ids)

    def _adjust_review_list_and_ids(self, reviews, costar_ids, pad_costar_id):
        reviews = reviews[:self.review_count]
        costar_ids = costar_ids[:self.review_count]

        pad_review = [self.pad_id] * self.review_length

        while len(reviews) < self.review_count:
            reviews.append(pad_review)
            costar_ids.append(pad_costar_id)

        fixed_reviews = []
        for review in reviews:
            review = review[:self.review_length]
            review = review + [self.pad_id] * (self.review_length - len(review))
            fixed_reviews.append(review)

        return fixed_reviews, costar_ids

    def _review2id(self, review):
        if not isinstance(review, str):
            return []

        wids = []
        for word in review.split():
            wids.append(self.word_dict.get(word, self.pad_id))

        return wids