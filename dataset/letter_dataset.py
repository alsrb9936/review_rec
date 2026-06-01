import torch
from dataset.base_dataset import BaseDataset
from omegaconf import DictConfig
import pandas as pd


class LetterDataset(BaseDataset):
    def __init__(self, df: pd.DataFrame, cfg: DictConfig, split: str = "train"):
        super().__init__(df, cfg, split)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
        }
