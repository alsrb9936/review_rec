import json
import os
import pickle
import re
import numpy as np
import pandas as pd

from omegaconf import DictConfig, open_dict, OmegaConf
from nltk.tokenize import WordPunctTokenizer


def bert_preprocess(train_df, valid_df, test_df, cfg):
    """
    Preprocess the data for BERT-based models.

    Args:
        train_df (pd.DataFrame): The training data.
        valid_df (pd.DataFrame): The validation data.
        test_df (pd.DataFrame): The test data.
        cfg (DictConfig): The configuration object containing preprocessing parameters.
    Returns:
        None
    """
    print("Starting BERT preprocessing...")
    # Implement BERT-specific preprocessing steps here
    train_df["clean_review"] = bert_clean_review(train_df)
    breakpoint()

def bert_clean_review(train_df):
    def clean_one(review):
        string = re.sub(r"[^A-Za-z0-9',.!;?()]", " ", review)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r"!+", " ! ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"\\", " \\ ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)

        string = re.sub(r"\s{2,}", " ", string)
        string = re.sub(r"(\.|\s){7,}", " ... ", string)
        string = re.sub(r"(?<= )(\w \. )+(\w \.)", lambda x: x.group().replace(" ", ""), string)

        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'m", " \'m", string)
        string = re.sub(r"\'ll", " \'ll", string)

        string = re.sub(r"(?!(('(?=s\b))|('(?=ve\b))|('(?=re\b))|('(?=d\b))|('(?=ll\b))|('(?=m\b))|((?<=n\b)'(?=t\b))))'", " ", string)

        string = re.sub(' 0 ', ' zero ', string)
        string = re.sub(' 1 ', ' one ', string)
        string = re.sub(' 2 ', ' two ', string)
        string = re.sub(' 3 ', ' three ', string)
        string = re.sub(' 4 ', ' four ', string)
        string = re.sub(' 5 ', ' five ', string)
        string = re.sub(' 6 ', ' six ', string)
        string = re.sub(' 7 ', ' seven ', string)
        string = re.sub(' 8 ', ' eight ', string)
        string = re.sub(' 9 ', ' nine ', string)

        string = re.sub(r"\s{2,}", " ", string)
        string = string.strip().lower()
        return string

    return train_df["review_text"].apply(clean_one) 
