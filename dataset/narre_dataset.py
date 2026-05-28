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
    def __init__(self, df, cfg, word_dict, split="train"):
        super().__init__()

        self.df = df.copy().reset_index(drop=True)
        self.word_dict = word_dict

        self.review_length = int(cfg.data.review_length)
        self.review_count = int(cfg.data.review_count)
        self.pad_id = int(cfg.data.pad_id)

        self.lowest_r_count = int(cfg.data.lowest_review_count)

        # valid/test에서는 현재 예측 대상 user-item pair의 review 제거 권장
        self.retain_rui = False
        if split == "train":
            self.retain_rui = bool(cfg.data.retain_rui)

        self.df["review_text"] = self.df["review_text"].apply(self._review2id)

        self.sparse_idx = set()

        self.user_reviews, self.user_review_item_ids = self._get_reviews_and_ids(
            self.df,
            lead="user_id",
            costar="item_id",
        )

        self.item_reviews, self.item_review_user_ids = self._get_reviews_and_ids(
            self.df,
            lead="item_id",
            costar="user_id",
        )

        user_ids = torch.tensor(self.df["user_id"].values, dtype=torch.long)
        item_ids = torch.tensor(self.df["item_id"].values, dtype=torch.long)
        ratings = torch.tensor(self.df["rating"].values, dtype=torch.float32).view(-1, 1)

        keep_idx = [idx for idx in range(len(self.df)) if idx not in self.sparse_idx]

        self.user_ids = user_ids[keep_idx]
        self.item_ids = item_ids[keep_idx]
        self.ratings = ratings[keep_idx]

        self.user_reviews = self.user_reviews[keep_idx]
        self.user_review_item_ids = self.user_review_item_ids[keep_idx]

        self.item_reviews = self.item_reviews[keep_idx]
        self.item_review_user_ids = self.item_review_user_ids[keep_idx]

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

    def _get_reviews_and_ids(self, df, lead, costar):
        reviews_by_lead = dict(list(df[[costar, "review_text"]].groupby(df[lead])))

        all_reviews = []
        all_costar_ids = []

        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            group = reviews_by_lead[lead_id]

            if self.retain_rui:
                reviews = group["review_text"].to_list()
                costar_ids = group[costar].to_list()
            else:
                filtered = group[group[costar] != costar_id]
                reviews = filtered["review_text"].to_list()
                costar_ids = filtered[costar].to_list()

            if len(reviews) < self.lowest_r_count:
                self.sparse_idx.add(idx)

            reviews, costar_ids = self._adjust_review_list_and_ids(
                reviews,
                costar_ids,
            )

            all_reviews.append(reviews)
            all_costar_ids.append(costar_ids)

        return torch.LongTensor(all_reviews), torch.LongTensor(all_costar_ids)

    def _adjust_review_list_and_ids(self, reviews, costar_ids):
        reviews = reviews[:self.review_count]
        costar_ids = costar_ids[:self.review_count]

        pad_review = [self.pad_id] * self.review_length

        while len(reviews) < self.review_count:
            reviews.append(pad_review)
            costar_ids.append(0)

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
            wids.append(self.word_dict.get(word, self.unk_id))

        return wids