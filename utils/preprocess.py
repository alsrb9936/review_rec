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
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk.tokenize import WordPunctTokenizer

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

def clean_review(cfg, review_series):
    with open(cfg.data.stopword_file, "r", encoding="utf-8") as f:
        stop_words = set(line.strip() for line in f if line.strip())

    with open(cfg.data.punctuation_file, "r", encoding="utf-8") as f:
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

    return review_series.apply(clean_one)

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

def google_load_embedding(cfg):
    """
    Load GoogleNews word2vec bin file and add <pad> vector.
    """
    word2vec_file = cfg.data.word_embedding_file
    pad_word = "<pad>"
    word_dim = int(cfg.data.word_dim)

    word_vec = KeyedVectors.load_word2vec_format(
        word2vec_file,
        binary=True,
    )

    if pad_word not in word_vec.key_to_index:
        word_vec.add_vector(
            pad_word,
            np.zeros(word_dim, dtype=np.float32),
        )

    pad_id = word_vec.key_to_index[pad_word]

    cfg.data.pad_id = int(pad_id)
    cfg.data.vocab_size = len(word_vec.key_to_index)

    word_dict = word_vec.key_to_index

    return word_vec, word_dict