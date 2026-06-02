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
    output_dir = cfg.data.output_path
    output_dir = os.path.join(output_dir, cfg.data.dataset, "glove")
    
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
    
    pad_id = 0
    word_emb, word_dict = glove_load_embedding(cfg)

    for frame in (train_df):
        frame["review_ids"] = frame["clean_review"].apply(
            lambda x: review2id(x, word_dict, pad_id)
        )
    
    # 5. train 기준 entity docs 생성
    all_df = pd.concat([train_df, valid_df, test_df], axis=0)
    num_users = int(all_df["user_id"].max()) + 1
    num_items = int(all_df["item_id"].max()) + 1
    del all_df

    user_doc, item_doc = build_entity_docs_from_train(
        train_df=train_df,
        num_users=num_users,
        num_items=num_items,
        review_count=max_review_count,
        review_length=max_review_len,
        pad_id=pad_id,
    )

    # split-level docs
    save_split_docs("train", train_df, user_doc, item_doc, output_dir, review_length=max_review_len, pad_id=pad_id )
    save_split_docs("valid", valid_df, user_doc, item_doc, output_dir, review_length=max_review_len, pad_id=pad_id )
    save_split_docs("test", test_df, user_doc, item_doc, output_dir, review_length=max_review_len, pad_id=pad_id )

def build_entity_docs_from_train(
    train_df,
    num_users,
    num_items,
    review_count,
    review_length,
    pad_id=0,
):
    """
    train_df만으로 user_doc, item_doc 생성.

    user_doc[user_id] -> 해당 user의 train reviews
    item_doc[item_id] -> 해당 item의 train reviews
    """
    user_doc = np.full(
        (num_users, review_count, review_length),
        fill_value=pad_id,
        dtype=np.int64,
    )
    item_doc = np.full(
        (num_items, review_count, review_length),
        fill_value=pad_id,
        dtype=np.int64,
    )

    user_groups = train_df.groupby("user_id")["review_ids"].apply(list).to_dict()
    item_groups = train_df.groupby("item_id")["review_ids"].apply(list).to_dict()

    for user_id, review_id_list in user_groups.items():
        user_doc[int(user_id)] = encode_doc(
            review_id_list,
            review_count=review_count,
            review_length=review_length,
            pad_id=pad_id,
        )

    for item_id, review_id_list in item_groups.items():
        item_doc[int(item_id)] = encode_doc(
            review_id_list,
            review_count=review_count,
            review_length=review_length,
            pad_id=pad_id,
        )

    return user_doc, item_doc   

def save_split_docs(split_name, df, user_doc, item_doc, save_dir, review_length, pad_id=0):
    """
    split별 row 순서에 맞춘 user_doc/item_doc 저장.
    valid/test도 user_doc, item_doc 자체는 train_df로 만든 것을 lookup한다.
    """
    user_ids = df["user_id"].astype(np.int64).to_numpy()
    item_ids = df["item_id"].astype(np.int64).to_numpy()
    ratings = df["rating"].astype(np.float32).to_numpy()

    np.save(os.path.join(save_dir, f"{split_name}_user_id.npy"), user_ids)
    np.save(os.path.join(save_dir, f"{split_name}_item_id.npy"), item_ids)
    np.save(os.path.join(save_dir, f"{split_name}_rating.npy"), ratings)

    np.save(os.path.join(save_dir, f"{split_name}_user_doc.npy"), user_doc[user_ids])
    np.save(os.path.join(save_dir, f"{split_name}_item_doc.npy"), item_doc[item_ids])

    # TransNet에서 target review를 쓸 수 있게 split별 현재 interaction review도 저장.
    target_doc = encode_target_reviews(df, review_length=review_length, pad_id=pad_id)
    np.save(os.path.join(save_dir, f"{split_name}_target_doc.npy"), target_doc)

def percentile_cap(values):
    percent=0.85
    values = [int(v) for v in values if pd.notna(v) and int(v) > 0]

    values = np.sort(values)
    idx = int(np.ceil(len(values) * percent)) - 1
    idx = max(0, min(idx, len(values) - 1))

    return int(values[idx])

def pad_or_truncate(ids, max_len, pad_id=0):
    ids = ids[:max_len]
    ids = ids + [pad_id] * (max_len - len(ids))
    return ids

def encode_doc(review_id_list, review_count, review_length, pad_id=0):
    """
    review_id_list: list[list[int]]
        한 user 또는 한 item이 가진 review들의 token id list.

    return:
        np.ndarray, shape = [review_count, review_length]
    """
    doc = []

    for ids in review_id_list[:review_count]:
        doc.append(pad_or_truncate(ids, review_length, pad_id))

    pad_review = [pad_id] * review_length
    while len(doc) < review_count:
        doc.append(pad_review)

    return np.asarray(doc, dtype=np.int64)

def encode_target_reviews(df, review_length, pad_id=0):
    target = np.full(
        (len(df), review_length),
        fill_value=pad_id,
        dtype=np.int64,
    )

    for row_idx, ids in enumerate(df["review_ids"].tolist()):
        target[row_idx] = np.asarray(
            pad_or_truncate(ids, review_length, pad_id),
            dtype=np.int64,
        )

    return target

def review2id(review, word_dict, pad_id=0):
    if not isinstance(review, str):
        return []

    ids = []
    for word in review.split():
        ids.append(word_dict.get(word, pad_id))
    return ids

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


