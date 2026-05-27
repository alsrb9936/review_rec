import ast
import json
import os
import re
import pickle
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from gensim.models import Word2Vec

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def _load_language_model(model_name: str, gpu_id: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embedding_batch(model_name: str, texts: Sequence[str], batch_size: int = 32, gpu_id: int = 0):
    tokenizer, model, device = _load_language_model(model_name, gpu_id)

    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for start in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Getting Embedding", unit="batch"):
        batch = list(texts[start : start + batch_size])
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        embeddings.extend(sentence_embeddings.cpu().tolist())

    return embeddings


def _parse_embedding_string(value: str) -> list[float]:
    text = value.strip()
    if not text:
        raise ValueError("Review embedding string is empty.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)

    if not isinstance(parsed, (list, tuple)):
        raise ValueError("Review embedding must deserialize to a 1D sequence.")
    return [float(item) for item in parsed]


def normalize_review_embedding(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().flatten().to(dtype=torch.float32)
        return tensor.tolist()
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if isinstance(value, str):
        return _parse_embedding_string(value)
    raise TypeError(f"Unsupported review embedding type: {type(value)!r}")


def tokenize_review(text):
    if not isinstance(text, str):
        return []
    string = re.sub(r"[^A-Za-z]", " ", text)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return text.strip().lower().split()


def build_word2idx(tokenized_reviews, max_vocab=50000, min_count=1):
    counter = Counter()
    for tokens in tokenized_reviews:
        counter.update(tokens)

    word2idx = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for word, count in counter.most_common(max_vocab):
        if count < min_count:
            continue
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    return word2idx


def encode_tokens(tokens, word2idx):
    unk_id = word2idx[UNK_TOKEN]
    return [word2idx.get(token, unk_id) for token in tokens]


def build_user_item_review_bank(df):
    user_reviews = defaultdict(list)
    item_reviews = defaultdict(list)
    pair_pos = {}

    for row in df.itertuples(index=False):
        user_id = int(row.user_id)
        item_id = int(row.item_id)
        review_ids = list(row.review_ids)

        user_pos = len(user_reviews[user_id])
        item_pos = len(item_reviews[item_id])

        user_reviews[user_id].append(review_ids)
        item_reviews[item_id].append(review_ids)
        pair_pos[(user_id, item_id)] = (user_pos, item_pos)

    return dict(user_reviews), dict(item_reviews), pair_pos


def train_gensim_word2vec(train_tokens, word2idx, cfg):
    embedding_dim = int(cfg.w2v.embedding_dim)

    model = Word2Vec(
        sentences=train_tokens,
        vector_size=embedding_dim,
        window=int(cfg.w2v.window),
        min_count=int(cfg.w2v.min_count),
        workers=int(cfg.w2v.workers),
        sg=int(cfg.w2v.sg),
        negative=int(cfg.w2v.negative),
        epochs=int(cfg.w2v.epochs),
        seed=int(cfg.experiment.seed),
    )

    embedding = np.random.normal(
        loc=0.0,
        scale=0.01,
        size=(len(word2idx), embedding_dim),
    ).astype(np.float32)

    embedding[int(cfg.data.pad_id)] = np.zeros(embedding_dim, dtype=np.float32)

    for word, idx in word2idx.items():
        if word in model.wv:
            embedding[idx] = model.wv[word]

    return embedding


def _save_cache(resource_dir, word_embedding, word2idx, user_reviews, item_reviews,
                pair_pos, train_df, valid_df, test_df):
    np.save(os.path.join(resource_dir, "word_embedding.npy"), word_embedding)
    with open(os.path.join(resource_dir, "word2idx.pkl"), "wb") as f:
        pickle.dump(word2idx, f)
    with open(os.path.join(resource_dir, "user_reviews.pkl"), "wb") as f:
        pickle.dump(user_reviews, f)
    with open(os.path.join(resource_dir, "item_reviews.pkl"), "wb") as f:
        pickle.dump(item_reviews, f)
    with open(os.path.join(resource_dir, "pair_pos.pkl"), "wb") as f:
        pickle.dump(pair_pos, f)
    train_df.to_pickle(os.path.join(resource_dir, "train_df.pkl"))
    valid_df.to_pickle(os.path.join(resource_dir, "valid_df.pkl"))
    test_df.to_pickle(os.path.join(resource_dir, "test_df.pkl"))


def _load_cache(resource_dir):
    word_embedding_path = os.path.join(resource_dir, "word_embedding.npy")
    word_embedding = np.load(word_embedding_path)
    with open(os.path.join(resource_dir, "word2idx.pkl"), "rb") as f:
        word2idx = pickle.load(f)
    with open(os.path.join(resource_dir, "user_reviews.pkl"), "rb") as f:
        user_reviews = pickle.load(f)
    with open(os.path.join(resource_dir, "item_reviews.pkl"), "rb") as f:
        item_reviews = pickle.load(f)
    with open(os.path.join(resource_dir, "pair_pos.pkl"), "rb") as f:
        pair_pos = pickle.load(f)
    train_df = pd.read_pickle(os.path.join(resource_dir, "train_df.pkl"))
    valid_df = pd.read_pickle(os.path.join(resource_dir, "valid_df.pkl"))
    test_df = pd.read_pickle(os.path.join(resource_dir, "test_df.pkl"))
    return {
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "word2idx": word2idx,
        "user_reviews": user_reviews,
        "item_reviews": item_reviews,
        "pair_pos": pair_pos,
        "word_embedding_path": word_embedding_path,
    }


def build_review_text_resources(train_df, valid_df, test_df, cfg):

    resource_dir = os.path.join(
        cfg.data.cache_dir,
        "review_text_resources",
        cfg.data.dataset,
    )

    if not os.path.exists(resource_dir):
        os.makedirs(resource_dir, exist_ok=True)

    word_embedding_path = os.path.join(resource_dir, "word_embedding.npy")

    cache_files = [
        word_embedding_path,
        os.path.join(resource_dir, "word2idx.pkl"),
        os.path.join(resource_dir, "user_reviews.pkl"),
        os.path.join(resource_dir, "item_reviews.pkl"),
        os.path.join(resource_dir, "pair_pos.pkl"),
        os.path.join(resource_dir, "train_df.pkl"),
        os.path.join(resource_dir, "valid_df.pkl"),
        os.path.join(resource_dir, "test_df.pkl"),
    ]
    cache_exists = all(os.path.exists(f) for f in cache_files)

    if not cache_exists:
        if "review_text" not in train_df.columns:
            raise ValueError(
                "Review-text models require review_text column. "
                "Run with data.load_review_text=true and check .review file."
            )

        train_df = train_df.copy()
        valid_df = valid_df.copy()
        test_df = test_df.copy()

        train_df["tokens"] = train_df["review_text"].apply(tokenize_review)
        valid_df["tokens"] = valid_df["review_text"].apply(tokenize_review)
        test_df["tokens"] = test_df["review_text"].apply(tokenize_review)

        train_tokens = train_df["tokens"].tolist()

        word2idx = build_word2idx(
            train_tokens,
            max_vocab=int(cfg.w2v.max_vocab),
            min_count=int(cfg.w2v.min_count),
        )

        train_df["review_ids"] = train_df["tokens"].apply(lambda x: encode_tokens(x, word2idx))
        valid_df["review_ids"] = valid_df["tokens"].apply(lambda x: encode_tokens(x, word2idx))
        test_df["review_ids"] = test_df["tokens"].apply(lambda x: encode_tokens(x, word2idx))

        if cfg.data.retain_rui:
            all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
            user_reviews, item_reviews, pair_pos = build_user_item_review_bank(all_df)
        else:
            user_reviews, item_reviews, pair_pos = build_user_item_review_bank(train_df)

        word_embedding = train_gensim_word2vec(train_tokens, word2idx, cfg)

        _save_cache(
            resource_dir, word_embedding, word2idx, user_reviews, item_reviews,
            pair_pos, train_df, valid_df, test_df
        )

        return {
            "train_df": train_df,
            "valid_df": valid_df,
            "test_df": test_df,
            "word2idx": word2idx,
            "user_reviews": user_reviews,
            "item_reviews": item_reviews,
            "pair_pos": pair_pos,
            "word_embedding_path": word_embedding_path,
        }

    else:
        return _load_cache(resource_dir)
