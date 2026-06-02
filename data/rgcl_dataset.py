# dataset/rgcl_dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import ast


class RGCLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, split: str = "train"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.split = split

        self.user_ids = torch.tensor(self.df["user_id"].values, dtype=torch.long)
        self.item_ids = torch.tensor(self.df["item_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(self.df["rating"].values, dtype=torch.float32)

        # train에서 ED-MI를 쓸 때만 target review embedding 사용
        if split == "train":
            self.review_embedding = torch.from_numpy(
                self._stack_embeddings(self.df["review_embedding"].to_numpy())
            ).float()
        else:
            self.review_embedding = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        out = {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],
        }

        if self.split == "train":
            out["review_feat"] = self.review_embedding[idx]

        return out

    def _stack_embeddings(self, values):
        parsed = []
        for x in values:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            elif isinstance(x, np.ndarray):
                pass
            elif isinstance(x, list):
                x = np.asarray(x, dtype=np.float32)
            elif isinstance(x, str):
                x = np.asarray(ast.literal_eval(x), dtype=np.float32)
            else:
                raise TypeError(f"Unsupported embedding type: {type(x)}")
            parsed.append(np.asarray(x, dtype=np.float32))

        return np.stack(parsed).astype(np.float32)