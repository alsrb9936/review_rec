import os
import pickle
import random

import numpy as np
import pandas as pd
from omegaconf import DictConfig, open_dict
from sklearn.model_selection import train_test_split

def _apply_id_mapping(interactions: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    mapping_dir = os.path.join(cfg.data.root, cfg.data.dataset, "mappings")
    os.makedirs(mapping_dir, exist_ok=True)
    mapping_path = os.path.join(mapping_dir, "id_mappings.pkl")
    ####
    user_values = sorted(interactions["user_id"].unique())  
    item_values = sorted(interactions["item_id"].unique())
    user2idx = {value: idx for idx, value in enumerate(user_values)}
    item2idx = {value: idx for idx, value in enumerate(item_values)}

    mappings = {
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": {idx: value for value, idx in user2idx.items()},
        "idx2item": {idx: value for value, idx in item2idx.items()},
    }
    with open(mapping_path, "wb") as file:
        pickle.dump(mappings, file)

    remapped = interactions.copy()
    remapped["user_id"] = remapped["user_id"].map(user2idx).astype(int)
    remapped["item_id"] = remapped["item_id"].map(item2idx).astype(int)

    with open_dict(cfg):
        cfg.stats.num_users = len(user2idx)
        cfg.stats.num_items = len(item2idx)

    return remapped

def load_interaction_data(cfg: DictConfig) -> pd.DataFrame:
    data_root_path = f"{cfg.data.data_root}/{cfg.data.dataset}"
    review_path = f"{data_root_path}/{cfg.data.dataset}.review"

    interaction_path = f"{data_root_path}/{cfg.data.dataset}.inter"
    interactions = pd.read_csv(interaction_path, sep=cfg.data.separator)
    review_df = pd.read_csv(review_path, sep=cfg.data.separator)
    
    interactions.columns = [column.split(":")[0] for column in interactions.columns]
    review_df.columns = [column.split(":")[0] for column in review_df.columns]
    review_df = review_df.rename(columns={"reviewText": "review_text"})
    interactions["timestamp"] = pd.to_numeric(interactions["timestamp"], errors="coerce")
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")

    merge_keys = ["user_id", "item_id"]
    interactions = pd.merge(interactions, review_df[merge_keys + ["review_text"]], on=merge_keys, how="left")
    interactions["review_text"] = interactions["review_text"].fillna("").astype(str)
    interactions = interactions.drop(interactions[[not isinstance(x, str) or len(x) == 0 for x in interactions['review_text']]].index)  # erase null review_texts

    print(f"drop {len(review_df) - len(interactions)} rows without review text")
    print(f"Merged {len(interactions)} rows with review text")
    
    del review_df

        
    interactions = _apply_id_mapping(interactions, cfg)

    print(
        f"Loaded {len(interactions)} rows "
        f"for dataset='{cfg.data.dataset}' "
        f"with columns={list(interactions.columns)}"
    )
    return interactions


def split_by_ratio(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    random_state: int = 64,
    user_col: str = "user_id",
    item_col: str = "item_id",
):
    """
    Split dataframe into train / valid / test.

    Additional rule:
    - If a user or item appears only in valid/test but not in train,
      move those rows from valid/test to train.
    """

    test_ratio = 1.0 - train_ratio - valid_ratio


    # 1. Basic random split
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True
    )

    valid_ratio_in_temp = valid_ratio / (valid_ratio + test_ratio)

    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_ratio_in_temp,
        random_state=random_state,
        shuffle=True
    )

    # 2. Build user/item sets from train
    train_user_set = set(train_df[user_col].unique())
    train_item_set = set(train_df[item_col].unique())

    # 3. Find rows in valid/test whose user or item is not in train
    valid_cold_mask = (
        ~valid_df[user_col].isin(train_user_set)
        | ~valid_df[item_col].isin(train_item_set)
    )

    test_cold_mask = (
        ~test_df[user_col].isin(train_user_set)
        | ~test_df[item_col].isin(train_item_set)
    )

    # 4. Move those rows to train
    cold_valid_df = valid_df.loc[valid_cold_mask]
    cold_test_df = test_df.loc[test_cold_mask]

    train_df = pd.concat(
        [train_df, cold_valid_df, cold_test_df],
        axis=0
    )

    # 5. Remove them from valid/test
    valid_df = valid_df.loc[~valid_cold_mask]
    test_df = test_df.loc[~test_cold_mask]

    # 6. Reset indices
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )
