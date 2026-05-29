import torch
from torch.utils.data import Dataset
from typing import Optional
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import ast
class MyModelDataset(Dataset):
    def __init__(
            self, 
            df: pd.DataFrame, 
            cfg: DictConfig, 
            split: str = "train",
            history_df: Optional[pd.DataFrame] = None):
        """
        df:
            현재 Dataset의 target interactions.
            train이면 train_df, valid이면 valid_df, test이면 test_df.

        history_df:
            user/item review embedding mean pooling을 만들 source.
            train/valid/test 모두 보통 train_df를 넘긴다.

        retain_rui:
            train에서 현재 row의 review embedding을 user/item mean에 포함할지 여부.
            True  -> 현재 row review embedding 포함
            False -> 현재 row review embedding 제외, leave-one-out
        """
        super().__init__()
        if history_df is None:
            if split in ["valid", "test"]:
                raise ValueError(
                    "You have to put history_df=train_df in valid/test Dataset"
                )
            history_df = df

          # 원본 index를 보존해야 train에서 현재 row를 정확히 제외할 수 있음
        target_df = df.copy()
        target_df["_target_row_id"] = target_df.index
        target_df = target_df.reset_index(drop=True)

        history_df = history_df.copy()
        history_df["_history_row_id"] = history_df.index
        history_df = history_df.reset_index(drop=True)

 
        self.df = target_df
        self.history_df = history_df
        self.cfg = cfg
        self.split = split

        self.retain_rui = False
        if split == "train":
            self.retain_rui = bool(cfg.data.retain_rui)

        self.user_ids = torch.tensor(
            target_df["user_id"].values,
            dtype=torch.long,
        )
        self.item_ids = torch.tensor(
            target_df["item_id"].values,
            dtype=torch.long,
        )
        self.ratings = torch.tensor(
            target_df["rating"].values,
            dtype=torch.float32,
        ).view(-1, 1)

        # target_review_emb = self._stack_embeddings(
        #     target_df["review_embedding"].to_numpy()
        # )
        # self.review_embedding = torch.from_numpy(target_review_emb).float()

        history_review_emb = self._stack_embeddings(
            history_df["review_embedding"].to_numpy()
        )
        self.history_review_embedding = torch.from_numpy(history_review_emb).float()

        #train history 전체 평균
        self.global_review_embedding = self.history_review_embedding.mean(dim=0)

        exclude_current = split == "train" and not self.retain_rui

        self.user_review = self._get_group_mean_embedding(
            target_df=target_df,
            history_df=history_df,
            group_col="user_id",
            exclude_current=exclude_current,
        )

        self.item_review = self._get_group_mean_embedding(
            target_df=target_df,
            history_df=history_df,
            group_col="item_id",
            exclude_current=exclude_current,
        )


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "rating": self.ratings[idx],

            # 현재 interaction review embedding
            # rating prediction input으로 쓰면 leakage 가능성 있음
            # "review": self.review_embedding[idx],

            # train history 기반 user/item review profile
            "user_review": self.user_review[idx],
            "item_review": self.item_review[idx],
        }

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

    def _get_group_mean_embedding(
        self,
        target_df: pd.DataFrame,
        history_df: pd.DataFrame,
        group_col: str,
        exclude_current: bool,
    ):
        history_emb = self.history_review_embedding.detach().cpu().numpy()
        emb_dim = history_emb.shape[1]

        global_emb = self.global_review_embedding.detach().cpu().numpy()

        group_to_indices = history_df.groupby(group_col).indices

        pooled = np.zeros((len(target_df), emb_dim), dtype=np.float32)

        for idx, row in target_df.iterrows():
            group_id = row[group_col]

            # valid/test에서 train history에 없는 user/item
            if group_id not in group_to_indices:
                pooled[idx] = global_emb
                continue

            row_indices = np.asarray(group_to_indices[group_id], dtype=np.int64)

            # train에서 현재 interaction review embedding 제외
            if exclude_current:
                target_row_id = row["_target_row_id"]

                history_row_ids = history_df.iloc[row_indices][
                    "_history_row_id"
                ].to_numpy()

                keep_mask = history_row_ids != target_row_id
                row_indices = row_indices[keep_mask]

            # leave-one-out 후 남는 history가 없는 경우
            if len(row_indices) == 0:
                pooled[idx] = global_emb
            else:
                pooled[idx] = history_emb[row_indices].mean(axis=0)

        return torch.from_numpy(pooled).float()