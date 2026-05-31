import os
import pickle
import random
from typing import cast, Optional

import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, open_dict
from utils.preprocess import glove_load_embedding, google_load_embedding, clean_review, get_embedding_batch
from torch.utils.data import DataLoader
from dataset import DATASET_DICT
from sklearn.model_selection import train_test_split
from utils.graph import build_lightgcn_norm_adj
from utils.graph import build_rgcl_graph

REVIEW_TEXT_MODEL_NAMES = {"deepconn", "narre", "transnet"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


def _load_cached_review_embeddings(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".pt":
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, torch.Tensor):
            return [tensor.clone().detach() for tensor in loaded]
        return loaded
    if suffix == ".npy":
        loaded = np.load(path, allow_pickle=False)
        return [row for row in loaded]
    raise ValueError(f"Unsupported review embedding format: {path}")


def _embedding_cache_path(cfg: DictConfig, dataset: str, model_name: str) -> str:
    safe_model = model_name.split("/")[-1]
    embedding_path = cfg.data.get("cache_dir")
    if embedding_path is None:
        embedding_path = os.path.join(to_absolute_path(cfg.data.cache_dir), "embeddings")
    return os.path.join(to_absolute_path(embedding_path), dataset, f"{safe_model}.pt")


def _compute_review_embeddings_for_frame(df: pd.DataFrame, cfg: DictConfig, gpu_id: int = 0):
    review_embeddings = [None] * len(df)
    non_empty_idx = [
        i for i, text in enumerate(df["review_text"].tolist())
        if isinstance(text, str) and text.strip()
    ]
    if non_empty_idx:
        non_empty_texts = [df.iloc[i]["review_text"] for i in non_empty_idx]
        model_name = str(cfg.data.get("language_model", "sentence-transformers/all-MiniLM-L6-v2"))
        batch_size = int(cfg.data.get("embedding_batch_size", 16))
        predicted_embeddings = get_embedding_batch(
            model_name, non_empty_texts, batch_size=batch_size, gpu_id=gpu_id
        )
        for idx, emb in zip(non_empty_idx, predicted_embeddings):
            review_embeddings[idx] = emb
    return review_embeddings

def _attach_review_embeddings(df, cfg: DictConfig):
    dataset = str(cfg.data.dataset)
    model_name = str(cfg.data.get("language_model", "sentence-transformers/all-MiniLM-L6-v2"))
    cache_path = _embedding_cache_path(cfg, dataset, model_name)

    if os.path.exists(cache_path):
        print(f"Load cached review embeddings from {cache_path}")
        cached = _load_cached_review_embeddings(cache_path)
        df["review_embedding"] = cached
        return df
    else:
        print(f"Computing review embeddings with {model_name}...")
        gpu_id = int(cfg.experiment.device)
        df["review_embedding"] = _compute_review_embeddings_for_frame(df, cfg, gpu_id=gpu_id)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(df["review_embedding"].tolist(),cache_path)
        print(f"Saved review embeddings to {cache_path}")

        return df

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
    data_root_path = to_absolute_path(f"{cfg.data.root}/{cfg.data.dataset}")
    interaction_path = to_absolute_path(f"{data_root_path}/{cfg.data.dataset}.inter")
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

    wants_review_text = cfg.data.get("load_review_text", False)
    plm_embedding = cfg.data.get("plm_embedding", False)
    if wants_review_text:
        review_path = to_absolute_path(f"{data_root_path}/{cfg.data.dataset}.review")
        review_df = pd.read_csv(review_path, sep=cfg.data.separator)
        review_df = _rename_columns(review_df, cfg)

   
        merge_keys = ["user_id", "item_id"]
        interactions = pd.merge(interactions, review_df[merge_keys + ["review_text"]], on=merge_keys, how="left")
        interactions["review_text"] = interactions["review_text"].fillna("").astype(str)
        print(f"Merged {len(interactions)} rows with review text")
      
        if plm_embedding:
            interactions = _attach_review_embeddings(interactions, cfg)

        
    # interactions = interactions.sort_values("timestamp").reset_index(drop=True)
    interactions = _apply_id_mapping(interactions, cfg)

    print(
        f"Loaded {len(interactions)} rows "
        f"for dataset='{cfg.data.dataset}' "
        f"with columns={list(interactions.columns)}"
    )
    return interactions


def split_by_ratio(df, train_ratio=0.8, valid_ratio=0.1, random_state=42):
    test_ratio = 1.0 - train_ratio - valid_ratio

    if test_ratio <= 0:
        raise ValueError("train_ratio + valid_ratio must be smaller than 1.0")

    # 1차 split: train / temp(valid + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True
    )

    # temp 안에서 valid 비율 계산
    valid_ratio_in_temp = valid_ratio / (valid_ratio + test_ratio)

    # 2차 split: valid / test
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_ratio_in_temp,
        random_state=random_state,
        shuffle=True
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )

def get_dataloader(cfg: DictConfig):
    model_name = cfg.model_name.lower()
    interactions = load_interaction_data(cfg)
    dataset_cls = DATASET_DICT[model_name]
    
    if cfg.data.get("load_review_text", False):
        interactions = interactions.drop(interactions[[not isinstance(x, str) or len(x) == 0 for x in interactions['review_text']]].index)  # erase null review_texts

    if model_name in REVIEW_TEXT_MODEL_NAMES:
        interactions['review_text'] = clean_review(cfg, interactions['review_text'])
        train_df, valid_df, test_df = split_by_ratio(interactions, train_ratio=cfg.data.split.train_ratio, valid_ratio=cfg.data.split.valid_ratio, random_state=cfg.experiment.seed)
        if cfg.data.word_embedding_type == "glove":
            word_emb, word_dict = glove_load_embedding(cfg)
        elif cfg.data.word_embedding_type == "google":
            word_emb, word_dict = google_load_embedding(cfg)

        train_dataset = dataset_cls(
            train_df,
            cfg,
            word_dict,
            split="train",
            history_df=train_df,
        )
        valid_dataset = dataset_cls(
            valid_df,
            cfg,
            word_dict,
            split="valid",
            history_df=train_df,
        )
        test_dataset = dataset_cls(
            test_df,
            cfg,
            word_dict,
            split="test",
            history_df=train_df,
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch,
            shuffle=True,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=cfg.training.eval_batch,
            shuffle=False,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.training.eval_batch,
            shuffle=False,
        )

        return train_dataloader, valid_dataloader, test_dataloader, word_emb, word_dict
    
    elif model_name in [
        "mymodel",
    ]:
        interactions["review_text"] = clean_review(cfg, interactions["review_text"])

        train_df, valid_df, test_df = split_by_ratio(
            interactions,
            train_ratio=cfg.data.split.train_ratio,
            valid_ratio=cfg.data.split.valid_ratio,
            random_state=cfg.experiment.seed,
        )

        train_dataset = dataset_cls(train_df, cfg, split="train")
        valid_dataset = dataset_cls(valid_df, cfg, split="valid", history_df=train_df)
        test_dataset = dataset_cls(test_df, cfg, split="test", history_df=train_df)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch,
            shuffle=True,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=cfg.training.eval_batch,
            shuffle=False,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.training.eval_batch,
            shuffle=False,
        )

        norm_adj = build_lightgcn_norm_adj(
            train_df=train_df,
            num_users=cfg.stats.num_users,
            num_items=cfg.stats.num_items,
        )

        return train_dataloader, valid_dataloader, test_dataloader, norm_adj
    elif model_name in ["rgcl"]:
        interactions["review_text"] = clean_review(cfg, interactions["review_text"])

        train_df, valid_df, test_df = split_by_ratio(
            interactions,
            train_ratio=cfg.data.split.train_ratio,
            valid_ratio=cfg.data.split.valid_ratio,
            random_state=cfg.experiment.seed,
        )

        train_dataset = dataset_cls(train_df, cfg, split="train")
        valid_dataset = dataset_cls(valid_df, cfg, split="valid")
        test_dataset = dataset_cls(test_df, cfg, split="test")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch,
            shuffle=True,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=cfg.training.eval_batch,
            shuffle=False,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.training.eval_batch,
            shuffle=False,
        )

        rgcl_graph = build_rgcl_graph(
            train_df=train_df,
            num_users=cfg.stats.num_users,
            num_items=cfg.stats.num_items,
        )

        return train_dataloader, valid_dataloader, test_dataloader, rgcl_graph
    else:        
        train_df, valid_df, test_df = split_by_ratio(interactions, train_ratio=cfg.data.split.train_ratio, valid_ratio=cfg.data.split.valid_ratio, random_state=cfg.experiment.seed)

        train_dataset = dataset_cls(train_df, cfg, split="train")
        valid_dataset = dataset_cls(valid_df, cfg, split="valid")
        test_dataset = dataset_cls(test_df, cfg, split="test")

        train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.training.eval_batch, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.training.eval_batch, shuffle=False)


    return train_dataloader, valid_dataloader, test_dataloader