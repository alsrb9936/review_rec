import json
import os
import pickle
import random

import numpy as np
import pandas as pd

from utils.load_data import load_interaction_data, split_by_ratio
from utils.glove_pro import glove_preprocess
from utils.bert_pro import bert_preprocess

import hydra
from omegaconf import DictConfig, open_dict

def save_preprocessed_data(train_df, valid_df, test_df, output_path):
    """
    Save the preprocessed data to a specified output path.

    Args:
        train_df (pd.DataFrame): The preprocessed training data.
        valid_df (pd.DataFrame): The preprocessed validation data.
        test_df (pd.DataFrame): The preprocessed test data.
        output_path (str): The file path where the preprocessed data will be saved.
    """
    os.makedirs(output_path, exist_ok=True)
    train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(output_path, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(output_path, "test.csv"), index=False)

def load_preprocessed_data(output_path):
    """
    Load the preprocessed data from a specified output path.

    Args:
        output_path (str): The file path where the preprocessed data is saved.

    Returns:
        tuple: A tuple containing the training, validation, and test DataFrames.
    """
    train_df = pd.read_csv(os.path.join(output_path, "train.csv"))
    valid_df = pd.read_csv(os.path.join(output_path, "valid.csv"))
    test_df = pd.read_csv(os.path.join(output_path, "test.csv"))
    return train_df, valid_df, test_df

def save_stat(train_df, valid_df, test_df, output_path):
    """
    Save the statistics of the preprocessed data to a specified output path.

    Args:
        train_df (pd.DataFrame): The preprocessed training data.
        valid_df (pd.DataFrame): The preprocessed validation data.
        test_df (pd.DataFrame): The preprocessed test data.
        output_path (str): The file path where the statistics will be saved.
    """
    stats = {
        "train": {
            "num_samples": len(train_df),
            "num_users": train_df["user_id"].nunique(),
            "num_items": train_df["item_id"].nunique(),
        },
        "valid": {
            "num_samples": len(valid_df),
            "num_users": valid_df["user_id"].nunique(),
            "num_items": valid_df["item_id"].nunique(),
        },
        "test": {
            "num_samples": len(test_df),
            "num_users": test_df["user_id"].nunique(),
            "num_items": test_df["item_id"].nunique(),
        },
    }
    with open(os.path.join(output_path, "stats.json"), "w") as f:
        json.dump(stats, f)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to execute the data preprocessing steps.

    This function loads the interaction data, applies necessary preprocessing steps,
    and saves the preprocessed data to the specified output path.

    Returns:
        None
    """
    with open_dict(cfg):
        if "separator" not in cfg.data or cfg.data.separator is None:
            cfg.data.separator = "\t"
    # Save preprocessed data
    output_dir = os.path.join(cfg.data.root, cfg.data.dataset, "common")
    if os.path.exists(os.path.join(output_dir, "train.csv")):
        print(f"Preprocessed data already exists at {output_dir}. Skipping preprocessing.")
        train_df, valid_df, test_df = load_preprocessed_data(output_dir)
    else:
        interactions = load_interaction_data(cfg)
        train_df, valid_df, test_df = split_by_ratio(interactions, train_ratio=0.8, valid_ratio=0.1)
        os.makedirs(output_dir, exist_ok=True)
        save_preprocessed_data(train_df, valid_df, test_df, output_dir)
        save_stat(train_df, valid_df, test_df, output_dir)
    

    if cfg.data.type == "glove":
        glove_preprocess(train_df, valid_df, test_df, cfg)
    elif cfg.data.type == "bert":
        bert_preprocess(train_df, valid_df, test_df, cfg)
    elif cfg.data.type == "sentiment":
        # No additional preprocessing needed for sentiment-based model
        pass
    else:
        print(cfg.data.type)
        print("Invalid data type specified. Please choose either 'glove' or 'bert'.")

if __name__ == "__main__":
    main()
