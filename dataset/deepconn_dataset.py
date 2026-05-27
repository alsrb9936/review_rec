import torch
from dataset.base_dataset import BaseDataset
from omegaconf import DictConfig
import pandas as pd


class DeepCoNNDataset(BaseDataset):
    def __init__(self, df: pd.DataFrame, cfg: DictConfig, split: str = "train"):
        super().__init__(df, cfg, split)
        self.cfg = cfg
        self.review_length = cfg.data.review_length
        self.review_count = cfg.data.review_count
        self.lowest_r_count = cfg.data.lowest_r_count

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
            "user_reviews": self.user_reviews[idx],
            "item_reviews": self.item_reviews[idx],
        }

    def _get_review(self):
        self.user_ids = torch.tensor(self.df["user_id"].values, dtype=torch.long)
        self.item_ids = torch.tensor(self.df["item_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(self.df["rating"].values, dtype=torch.float)

        # Assuming review_embeddings is a list of lists (or a 2D array) in the DataFrame
        review_emb_list = self.df["review_embedding"].tolist()
        self.review_embeddings = torch.tensor(review_emb_list, dtype=torch.float)

    def _load_word_dict(self):
        # This method can be used to load a word dictionary if needed
        pass

    def _load_review_embeddings(self):
        # This method can be used to load review embeddings if they are stored separately
        pass

