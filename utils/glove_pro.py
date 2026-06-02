import json
import os
import pickle
import re
import numpy as np
import pandas as pd

from omegaconf import DictConfig, open_dict, OmegaConf
from nltk.tokenize import WordPunctTokenizer


def glove_preprocess(train_df, valid_df, test_df, cfg):
    """
    Preprocess the data for GloVe-based models.

    Args:
        train_df (pd.DataFrame): The training data.
        valid_df (pd.DataFrame): The validation data.
        test_df (pd.DataFrame): The test data.
        cfg (DictConfig): The configuration object containing preprocessing parameters.
    Returns:
        None
    """
    print("Starting GloVe preprocessing...")
    print("Cleaning and calculating max_len, max_count...")
    train_df["clean_review"] = glove_clean_review(train_df, cfg)
    review_lens = train_df["clean_review"].str.split().map(len)
    max_review_len = percentile_cap(review_lens)

    # 4. user/item 하나당 최대 review 개수
    user_review_counts = train_df.groupby("user_id").size()
    item_review_counts = train_df.groupby("item_id").size()

    max_user_review_count = percentile_cap(user_review_counts)
    max_item_review_count = percentile_cap(item_review_counts)
    max_review_count = max(max_user_review_count, max_item_review_count)
    print(f"Max review length: {max_review_len}")
    print(f"Max user review count: {max_user_review_count}")
    print(f"Max item review count: {max_item_review_count}")
    print(f"Max review count: {max_review_count}")
    


def pad_or_truncate_review(review_ids, review_length, pad_id):
    review_ids = review_ids[:review_length]
    review_ids = review_ids + [pad_id] * (review_length - len(review_ids))
    return review_ids

def glove_load_embedding(cfg):
    word2vec_file = cfg.data.word_embedding_file
    with open(word2vec_file, encoding='utf-8') as f:
        word_emb = list()
        word_dict = dict()
        word_emb.append([0])
        word_dict['<UNK>'] = 0
        for line in f.readlines():
            tokens = line.split(' ')
            word_emb.append([float(i) for i in tokens[1:]])
            word_dict[tokens[0]] = len(word_dict)
        word_emb[0] = [0] * len(word_emb[1])
    return word_emb, word_dict

def encode_review_list(reviews, word_dict, review_count, review_length, pad_id):
    """
    reviews: list[str]
        한 user 또는 한 item이 가진 clean_review 리스트

    return:
        shape = [review_count, review_length]
    """
    encoded_reviews = []

    # review 개수 자르기
    reviews = reviews[:review_count]

    for review in reviews:
        review_ids = review2id(review, word_dict, pad_id)
        review_ids = pad_or_truncate_review(review_ids, review_length, pad_id)
        encoded_reviews.append(review_ids)

    # review 개수 padding
    pad_review = [pad_id] * review_length
    while len(encoded_reviews) < review_count:
        encoded_reviews.append(pad_review)

    return encoded_reviews


def review2id(review, word_dict, pad_id):
    if not isinstance(review, str):
        return []

    wids = []
    for word in review.split():
        if word in word_dict:
            wids.append(word_dict[word])
        else:
            wids.append(pad_id)
    return wids

def percentile_cap(values):
    percent=0.85
    values = [int(v) for v in values if pd.notna(v) and int(v) > 0]

    values = np.sort(values)
    idx = int(np.ceil(len(values) * percent)) - 1
    idx = max(0, min(idx, len(values) - 1))

    return int(values[idx])

def glove_clean_review(train_df, cfg):
    with open(cfg.data.stopwords_path, "r", encoding="utf-8") as f:
        stop_words = set(line.strip() for line in f if line.strip())

    with open(cfg.data.punctuation_path, "r", encoding="utf-8") as f:
        punctuations = [line.strip() for line in f if line.strip()]

    tokenizer = WordPunctTokenizer()

    if punctuations:
        punct_pattern = re.compile("|".join(re.escape(p) for p in sorted(punctuations, key=len, reverse=True)))
    else:
        punct_pattern = None

    def clean_one(review):
        if pd.isna(review):
            return ""

        review = str(review).lower()

        if punct_pattern is not None:
            review = punct_pattern.sub(" ", review)

        tokens = tokenizer.tokenize(review)
        tokens = [word for word in tokens if word not in stop_words]

        return " ".join(tokens)

    return train_df["review_text"].apply(clean_one)


