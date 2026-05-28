import abc
import torch
from torch.utils.data import Dataset
import pandas as pd
from omegaconf import DictConfig


class BaseDataset(Dataset, abc.ABC):
    def __init__(self, df: pd.DataFrame, cfg: DictConfig, split: str = "train"):
        self.df = df
        self.cfg = cfg
        self.split = split

        self.user_ids = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.item_ids = torch.tensor(df["item_id"].values, dtype=torch.long)
        
        if "rating" in df.columns:
            self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)
        else:
            self.ratings = torch.zeros(len(df), dtype=torch.float32)

        self.has_review_embedding = "review_embedding" in df.columns
        self.has_review_text = "review_text" in df.columns

        if self.has_review_embedding:
            self.review_embeddings = self._stack_embeddings(df["review_embedding"].tolist())

    def _stack_embeddings(self, emb_list):
        valid = [e for e in emb_list if e is not None]
        if not valid:
            return None
        if isinstance(valid[0], torch.Tensor):
            return torch.stack(valid)
        return torch.tensor(valid, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    @abc.abstractmethod
    def __getitem__(self, idx):
        ...
