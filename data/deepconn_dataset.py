import torch
from torch.utils.data import Dataset
import pandas as pd
from omegaconf import DictConfig
from typing import Optional

class DeepCoNNDataset(Dataset):
    def __init__(self, cfg: DictConfig,split: str = "train",):
        super().__init__()
        self.cfg = cfg
        self.split = split



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

    def _load_data(self, split: str):
        pass