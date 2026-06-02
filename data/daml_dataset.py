from typing import List, Optional

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


class DAMLDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: DictConfig,
        word_dict: dict[str, int],
        split: str = "train",
        history_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.word_dict = word_dict
        self.doc_len = int(cfg.data.doc_len)
        self.pad_id = int(cfg.data.pad_id)
        self.retain_rui = split == "train" and bool(cfg.data.retain_rui)

        self.df = df.copy().reset_index(drop=True)
        if history_df is None:
            history_df = df
        self.history_df = history_df.copy().reset_index(drop=True)

        self.df["review_text"] = self.df["review_text"].fillna("").astype(str)
        self.history_df["review_text"] = self.history_df["review_text"].fillna("").astype(str)

        self.user_docs = self._build_docs(
            target_df=self.df,
            history_df=self.history_df,
            lead="user_id",
            costar="item_id",
        )
        self.item_docs = self._build_docs(
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
            "user_doc": self.user_docs[idx],
            "item_doc": self.item_docs[idx],
        }

    def _build_docs(self, target_df: pd.DataFrame, history_df: pd.DataFrame, lead: str, costar: str):
        reviews_by_lead = {
            lead_id: group[[costar, "review_text"]]
            for lead_id, group in history_df.groupby(lead)
        }

        docs = []
        for lead_id, costar_id in zip(target_df[lead], target_df[costar]):
            group = reviews_by_lead.get(lead_id)
            if group is None:
                reviews: List[str] = []
            elif self.retain_rui:
                reviews = list(group["review_text"])
            else:
                reviews = list(group.loc[group[costar] != costar_id, "review_text"])

            docs.append(self._reviews_to_doc(reviews))

        return torch.LongTensor(docs)

    def _reviews_to_doc(self, reviews: List[str]):
        token_ids: List[int] = []
        for review_idx, review in enumerate(reviews):
            if review_idx > 0:
                token_ids.append(self.word_dict.get("<sep>", self.pad_id))
            token_ids.extend(self._review_to_ids(review))

        if len(token_ids) < self.doc_len:
            token_ids.extend([self.pad_id] * (self.doc_len - len(token_ids)))
        else:
            token_ids = token_ids[: self.doc_len]

        return token_ids

    def _review_to_ids(self, review: str):
        return [self.word_dict.get(word, self.pad_id) for word in review.split()]
