import os
import pickle
import random
from typing import cast, Optional

import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, open_dict
from utils.preprocess import build_review_text_resources
from torch.utils.data import DataLoader

from dataset import DATASET_DICT

REVIEW_TEXT_MODEL_NAMES = {"deepconn", "narre", "transnet"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _interaction_file_path(cfg: DictConfig) -> str:
    if cfg.data.interaction_file:
        return to_absolute_path(cfg.data.interaction_file)
    dataset_name = cfg.data.dataset
    return to_absolute_path(os.path.join(cfg.data.root, dataset_name, f"{dataset_name}.inter"))


def _review_file_path(cfg: DictConfig) -> Optional[str]:
    if cfg.data.review_file:
        return to_absolute_path(cfg.data.review_file)
    dataset_name = cfg.data.dataset
    review_path = to_absolute_path(os.path.join(cfg.data.root, dataset_name, f"{dataset_name}.review"))
    if os.path.exists(review_path):
        return review_path
    return None


def _rename_columns(frame: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [column.split(":")[0] for column in frame.columns]

    columns = cfg.data.columns
    rename_map = {}

    if columns.user_id in frame.columns:
        rename_map[columns.user_id] = "user_id"
    if columns.item_id in frame.columns:
        rename_map[columns.item_id] = "item_id"
    if columns.timestamp in frame.columns:
        rename_map[columns.timestamp] = "timestamp"
    if columns.rating in frame.columns:
        rename_map[columns.rating] = "rating"
    if columns.review_text in frame.columns:
        rename_map[columns.review_text] = "review_text"

    if rename_map:
        frame = frame.rename(columns=rename_map)

    return frame


def _apply_id_mapping(interactions: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    output_dir = to_absolute_path(cfg.experiment.save_dir)
    mapping_dir = os.path.join(output_dir, "mappings", cfg.data.dataset)
    os.makedirs(mapping_dir, exist_ok=True)
    mapping_path = os.path.join(mapping_dir, "id_mappings.pkl")

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
    interaction_path = _interaction_file_path(cfg)
    print(f"Load interaction data from {interaction_path}")

    interactions = pd.read_csv(interaction_path, sep=cfg.data.separator)
    interactions = _rename_columns(interactions, cfg)

    required_cols = ["user_id", "item_id", "timestamp"]
    missing = [col for col in required_cols if col not in interactions.columns]
    if missing:
        raise ValueError(f"Missing required columns in interaction data: {missing}")

    interactions["timestamp"] = pd.to_numeric(interactions["timestamp"], errors="coerce")
    if interactions["timestamp"].isna().any():
        raise ValueError("Failed to parse one or more timestamp values.")

    if "rating" in interactions.columns:
        interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")

    if cfg.data.get("load_review_text", False):
        review_path = _review_file_path(cfg)
        if review_path:
            print(f"Merging review text from {review_path}")
            review_df = pd.read_csv(review_path, sep=cfg.data.separator)
            review_df = _rename_columns(review_df, cfg)
            if "review_text" in review_df.columns:
                merge_keys = ["user_id", "item_id"]
                interactions = pd.merge(interactions, review_df[merge_keys + ["review_text"]], on=merge_keys, how="left")
                interactions["review_text"] = interactions["review_text"].fillna("").astype(str)
                print(f"Merged {len(interactions)} rows with review text")
        else:
            print("Review file not found, skipping review text merge")

    interactions = interactions.sort_values("timestamp").reset_index(drop=True)
    interactions = _apply_id_mapping(interactions, cfg)

    print(
        f"Loaded {len(interactions)} rows "
        f"for dataset='{cfg.data.dataset}' "
        f"with columns={list(interactions.columns)}"
    )
    return interactions


def split_by_ratio(df, train_ratio=0.8, valid_ratio=0.1, random_state=42):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1.")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be less than 1.")

    if df.empty:
        empty = df.iloc[0:0].copy()
        return empty, empty.copy(), empty.copy()

    shuffled = cast(pd.DataFrame, df.sample(frac=1.0, random_state=random_state).reset_index(drop=True))
    total_count = len(shuffled)

    if total_count == 1:
        return shuffled.copy(), shuffled.iloc[0:0].copy(), shuffled.iloc[0:0].copy()

    if total_count == 2:
        return shuffled.iloc[:1].reset_index(drop=True), shuffled.iloc[1:].reset_index(drop=True), shuffled.iloc[0:0].copy()

    train_count = max(1, int(round(total_count * train_ratio)))
    valid_count = max(1, int(round(total_count * valid_ratio)))

    if train_count + valid_count >= total_count:
        overflow = train_count + valid_count - (total_count - 1)
        if valid_count > 1:
            reduction = min(overflow, valid_count - 1)
            valid_count -= reduction
            overflow -= reduction
        if overflow > 0 and train_count > 1:
            train_count -= min(overflow, train_count - 1)

    test_count = total_count - train_count - valid_count
    if test_count <= 0:
        test_count = 1
        if valid_count > 1:
            valid_count -= 1
        else:
            train_count -= 1

    train_end = train_count
    valid_end = train_end + valid_count

    train_df = shuffled.iloc[:train_end].reset_index(drop=True)
    valid_df = shuffled.iloc[train_end:valid_end].reset_index(drop=True)
    test_df = shuffled.iloc[valid_end:].reset_index(drop=True)

    return train_df, valid_df, test_df


def get_dataloader(train_df, valid_df, test_df, cfg: DictConfig):
    model_name = cfg.model_name.lower()

    if model_name not in DATASET_DICT:
        raise ValueError(f"Dataset class for model '{model_name}' is not registered in DATASET_DICT.")

    dataset_cls = DATASET_DICT[model_name]

    if model_name in REVIEW_TEXT_MODEL_NAMES:
        resources = build_review_text_resources(train_df, valid_df, test_df, cfg)

        with open_dict(cfg):
            cfg.data.word_embedding_path = resources["word_embedding_path"]
            cfg.stats.num_words = len(resources["word2idx"])

        train_dataset = dataset_cls(
            resources["train_df"],
            cfg,
            split="train",
            user_review_bank=resources["user_reviews"],
            item_review_bank=resources["item_reviews"],
            pair_pos=resources["pair_pos"],
        )

        valid_dataset = dataset_cls(
            resources["valid_df"],
            cfg,
            split="valid",
            user_review_bank=resources["user_reviews"],
            item_review_bank=resources["item_reviews"],
        )

        test_dataset = dataset_cls(
            resources["test_df"],
            cfg,
            split="test",
            user_review_bank=resources["user_reviews"],
            item_review_bank=resources["item_reviews"],
        )

    else:
        train_dataset = dataset_cls(train_df, cfg, split="train")
        valid_dataset = dataset_cls(valid_df, cfg, split="valid")
        test_dataset = dataset_cls(test_df, cfg, split="test")

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.training.eval_batch, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.training.eval_batch, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader