import ast
import json
import os
import re
import pickle
from collections.abc import Sequence, Counter, defaultdict
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



def tokenize_review(text: str) -> list[str]:
    if not isinstance(text, str):
        return []

    text = re.sub(r"[^A-Za-z0-9가-힣]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower().split()


def build_word2idx(tokenized_reviews, max_vocab: int = 50000, min_count: int = 1):
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


def build_user_item_review_bank(train_df: pd.DataFrame):
    user_reviews = defaultdict(list)
    item_reviews = defaultdict(list)
    pair_pos = {}

    for row in train_df.itertuples(index=False):
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
        seed=int(cfg.seed),
    )

    embedding = np.random.normal(
        loc=0.0,
        scale=0.01,
        size=(len(word2idx), embedding_dim),
    ).astype(np.float32)

    embedding[0] = np.zeros(embedding_dim, dtype=np.float32)

    for word, idx in word2idx.items():
        if word in model.wv:
            embedding[idx] = model.wv[word]

    return embedding


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_deepconn_resources(train_df, valid_df, test_df, cfg):
    """
    DeepCoNN용 tokenization, word2idx, Word2Vec, user/item review bank를 생성한다.
    train_df 기준으로만 vocab, Word2Vec, review bank를 만든다.
    """
    output_dir = os.path.abspath(cfg.experiment.save_dir)
    resource_dir = os.path.join(output_dir, "deepconn_resources", cfg.data.dataset)
    os.makedirs(resource_dir, exist_ok=True)

    word2idx_path = os.path.join(resource_dir, "word2idx.pkl")
    user_reviews_path = os.path.join(resource_dir, "user_reviews.pkl")
    item_reviews_path = os.path.join(resource_dir, "item_reviews.pkl")
    pair_pos_path = os.path.join(resource_dir, "pair_pos.pkl")
    word_embedding_path = os.path.join(resource_dir, "word_embedding.npy")

    # 이미 만들어져 있으면 재사용
    if (
        os.path.exists(word2idx_path)
        and os.path.exists(user_reviews_path)
        and os.path.exists(item_reviews_path)
        and os.path.exists(pair_pos_path)
        and os.path.exists(word_embedding_path)
    ):
        word2idx = load_pickle(word2idx_path)
        user_reviews = load_pickle(user_reviews_path)
        item_reviews = load_pickle(item_reviews_path)
        pair_pos = load_pickle(pair_pos_path)
        word_embedding = np.load(word_embedding_path)

        return {
            "word2idx": word2idx,
            "user_reviews": user_reviews,
            "item_reviews": item_reviews,
            "pair_pos": pair_pos,
            "word_embedding": word_embedding,
            "word_embedding_path": word_embedding_path,
        }

    # review 컬럼명 확인
    if "review" not in train_df.columns:
        raise ValueError(
            "DeepCoNN requires a raw text column named 'review'. "
            "현재 df에 review 컬럼이 없습니다."
        )

    # 1. tokenize
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()

    train_df["tokens"] = train_df["review"].apply(tokenize_review)
    valid_df["tokens"] = valid_df["review"].apply(tokenize_review)
    test_df["tokens"] = test_df["review"].apply(tokenize_review)

    # 2. vocab은 train 기준
    train_tokens = train_df["tokens"].tolist()
    word2idx = build_word2idx(
        train_tokens,
        max_vocab=int(cfg.w2v.max_vocab),
        min_count=int(cfg.w2v.min_count),
    )

    # 3. review ids 변환
    train_df["review_ids"] = train_df["tokens"].apply(lambda x: encode_tokens(x, word2idx))
    valid_df["review_ids"] = valid_df["tokens"].apply(lambda x: encode_tokens(x, word2idx))
    test_df["review_ids"] = test_df["tokens"].apply(lambda x: encode_tokens(x, word2idx))

    # 4. train 기준 user/item review bank 생성
    user_reviews, item_reviews, pair_pos = build_user_item_review_bank(train_df)

    # 5. train 기준 Word2Vec 학습
    word_embedding = train_gensim_word2vec(train_tokens, word2idx, cfg)

    # 6. 저장
    save_pickle(word2idx, word2idx_path)
    save_pickle(user_reviews, user_reviews_path)
    save_pickle(item_reviews, item_reviews_path)
    save_pickle(pair_pos, pair_pos_path)
    np.save(word_embedding_path, word_embedding)

    return {
        "word2idx": word2idx,
        "user_reviews": user_reviews,
        "item_reviews": item_reviews,
        "pair_pos": pair_pos,
        "word_embedding": word_embedding,
        "word_embedding_path": word_embedding_path,
    }