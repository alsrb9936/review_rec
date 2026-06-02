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
    os.makedirs(output_dir, exist_ok=True)
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
    word_emb = np.asarray(word_emb, dtype=np.float32)
    train_df["review_ids"] = train_df["clean_review"].apply(lambda x: review2id(x, word_dict, pad_id))
    
    # 5. train 기준 entity docs 생성
    all_df = pd.concat([train_df, valid_df, test_df], axis=0)
    num_users = int(all_df["user_id"].max()) + 1
    num_items = int(all_df["item_id"].max()) + 1
    pad_user_id = num_users
    pad_item_id = num_items
    del all_df

    user_doc, item_doc = build_entity_docs_from_train(
        train_df=train_df,
        num_users=num_users,
        num_items=num_items,
        review_count=max_review_count,
        review_length=max_review_len,
        pad_id=pad_id,
    )
    user_doc_item_ids, item_doc_user_ids = build_entity_doc_ids_from_train(
        train_df=train_df,
        num_users=num_users,
        num_items=num_items,
        review_count=max_review_count,
        pad_user_id=pad_user_id,
        pad_item_id=pad_item_id,
    )
    train_target_doc = encode_target_reviews(train_df, review_length=max_review_len, pad_id=pad_id)
    train_target_doc_emb = word_emb[train_target_doc]
    np.save(os.path.join(output_dir, "train_target_doc.npy"), train_target_doc)
    np.save(os.path.join(output_dir, "train_target_doc_emb.npy"),train_target_doc_emb.astype(np.float32))

    print("Saving split docs...")
    print("Saving Train split docs...")
    save_split_docs("train", train_df, user_doc, item_doc, user_doc_item_ids, item_doc_user_ids, output_dir, review_length=max_review_len, pad_id=pad_id, word_emb=word_emb)
    print("Saving Valid split docs...")
    save_split_docs("valid", valid_df, user_doc, item_doc, user_doc_item_ids, item_doc_user_ids, output_dir, review_length=max_review_len, pad_id=pad_id, word_emb=word_emb)
    print("Saving Test split docs...")
    save_split_docs("test", test_df, user_doc, item_doc, user_doc_item_ids, item_doc_user_ids, output_dir, review_length=max_review_len, pad_id=pad_id, word_emb=word_emb)
    
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
    print("Building entity docs...")
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

    print("Entity docs built.")
    return user_doc, item_doc   

def save_split_docs(split_name, df, user_doc, item_doc, user_doc_item_ids, item_doc_user_ids, save_dir, review_length, pad_id=0, word_emb=None):
    """
    split별 row 순서에 맞춘 user_doc/item_doc 저장.
    valid/test도 user_doc, item_doc 자체는 train_df로 만든 것을 lookup한다.
    """
    user_ids = df["user_id"].astype(np.int64).to_numpy()
    item_ids = df["item_id"].astype(np.int64).to_numpy()
    ratings = df["rating"].astype(np.float32).to_numpy()

    # common
    np.save(os.path.join(save_dir, f"{split_name}_user_id.npy"), user_ids)
    np.save(os.path.join(save_dir, f"{split_name}_item_id.npy"), item_ids)
    np.save(os.path.join(save_dir, f"{split_name}_rating.npy"), ratings)

    # id doc
    user_doc_ids = user_doc[user_ids]
    item_doc_ids = item_doc[item_ids]

    np.save(os.path.join(save_dir, f"{split_name}_user_doc.npy"), user_doc_ids)
    np.save(os.path.join(save_dir, f"{split_name}_item_doc.npy"), item_doc_ids)


    if word_emb is None:
        raise ValueError("word_emb must be provided when save_doc_emb=True.")

    user_doc_emb = word_emb[user_doc_ids]
    item_doc_emb = word_emb[item_doc_ids]

    np.save(os.path.join(save_dir, f"{split_name}_user_doc_emb.npy"),user_doc_emb.astype(np.float32))
    np.save(os.path.join(save_dir, f"{split_name}_item_doc_emb.npy"),item_doc_emb.astype(np.float32))

    # NARRE용 doc id 저장
    np.save(os.path.join(save_dir, f"{split_name}_user_review_item_ids.npy"),user_doc_item_ids[user_ids])
    np.save(os.path.join(save_dir, f"{split_name}_item_review_user_ids.npy"),item_doc_user_ids[item_ids])

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

def encode_doc_ids(costar_id_list, review_count, pad_costar_id):
    costar_id_list = [int(x) for x in costar_id_list[:review_count]]

    while len(costar_id_list) < review_count:
        costar_id_list.append(pad_costar_id)

    return np.asarray(costar_id_list, dtype=np.int64)

def build_entity_doc_ids_from_train(
    train_df,
    num_users,
    num_items,
    review_count,
    pad_user_id,
    pad_item_id,
):
    """
    NARRE용 id doc 생성.

    user_doc_item_ids[user_id]
    = user_doc[user_id]에 들어간 각 review의 item_id

    item_doc_user_ids[item_id]
    = item_doc[item_id]에 들어간 각 review의 user_id
    """

    user_doc_item_ids = np.full(
        (num_users, review_count),
        fill_value=pad_item_id,
        dtype=np.int64,
    )

    item_doc_user_ids = np.full(
        (num_items, review_count),
        fill_value=pad_user_id,
        dtype=np.int64,
    )

    user_groups = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
    item_groups = train_df.groupby("item_id")["user_id"].apply(list).to_dict()

    for user_id, item_id_list in user_groups.items():
        user_doc_item_ids[int(user_id)] = encode_doc_ids(
            item_id_list,
            review_count=review_count,
            pad_costar_id=pad_item_id,
        )

    for item_id, user_id_list in item_groups.items():
        item_doc_user_ids[int(item_id)] = encode_doc_ids(
            user_id_list,
            review_count=review_count,
            pad_costar_id=pad_user_id,
        )

    return user_doc_item_ids, item_doc_user_ids


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
    word2vec_file = cfg.data.glove_path
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


