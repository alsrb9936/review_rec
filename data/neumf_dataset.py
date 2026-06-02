import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
import pandas as pd


class NeuMFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: DictConfig, split: str = "train"):
        super().__init__()
        self.df = df
        self.cfg = cfg
        self.split = split

        self.user_ids = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.item_ids = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
        }

