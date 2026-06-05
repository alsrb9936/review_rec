
import json
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_NAME = "cardiffnlp/twitter-roberta-large-topic-sentiment-latest"
POSITIVE_LABELS = {"positive", "strongly positive"}
NEGATIVE_LABELS = {"strongly negative", "negative", "negative or neutral"}
LABEL_ORDER = [
    "strongly negative",
    "negative",
    "negative or neutral",
    "positive",
    "strongly positive",
]


def sentiment_preprocess(train_df, valid_df, test_df, cfg):
    """Analyze test reviews and save sentiment consistency row IDs.

    The saved IDs are zero-based row indices aligned to ``common/test.csv`` and
    every model-specific ``test_*.npy`` artifact. Two views are written:

    - ``sentiment_pos``: ratings 1/2 are negative and 3/4/5 are positive.
    - ``sentiment_neg``: ratings 1/2/3 are negative and 4/5 are positive.

    Each view partitions every test row into exactly one of consistent or
    inconsistent, so ``len(consistent_id) + len(inconsistent_id) == len(test)``.
    """
    del train_df, valid_df

    output_root = os.path.join(cfg.data.root, cfg.data.dataset)
    pos_dir = os.path.join(output_root, "sentiment_pos")
    neg_dir = os.path.join(output_root, "sentiment_neg")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    test_df = test_df.copy().reset_index(drop=True)
    if "review_text" not in test_df.columns:
        raise ValueError("sentiment preprocessing requires a review_text column in test_df")

    model_name = str(cfg.get("sentiment", {}).get("model_name", DEFAULT_MODEL_NAME))
    batch_size = int(cfg.get("sentiment", {}).get("batch_size", 32))
    gpu_id = int(cfg.experiment.get("device", 0))

    texts = test_df["review_text"].fillna("").astype(str).tolist()
    labels, scores = predict_sentiments(
        model_name=model_name,
        texts=texts,
        batch_size=batch_size,
        gpu_id=gpu_id,
        num_classes=5,
    )

    ratings = test_df["rating"].astype(float).to_numpy()
    labels_arr = np.asarray(labels, dtype=object)
    row_ids = np.arange(len(test_df), dtype=np.int64)

    positive_mask = np.isin(labels_arr, list(POSITIVE_LABELS))
    negative_mask = np.isin(labels_arr, list(NEGATIVE_LABELS))

    pos_rating_positive_mask = ratings >= 3.0
    pos_rating_negative_mask = ratings <= 2.0
    neg_rating_positive_mask = ratings >= 4.0
    neg_rating_negative_mask = ratings <= 3.0

    pos_consistent_mask = (
        (positive_mask & pos_rating_positive_mask)
        | (negative_mask & pos_rating_negative_mask)
    )
    neg_consistent_mask = (
        (positive_mask & neg_rating_positive_mask)
        | (negative_mask & neg_rating_negative_mask)
    )
    pos_inconsistent_mask = ~pos_consistent_mask
    neg_inconsistent_mask = ~neg_consistent_mask

    _save_sentiment_view(
        pos_dir,
        row_ids[pos_consistent_mask],
        row_ids[pos_inconsistent_mask],
        "positive",
    )
    _save_sentiment_view(
        neg_dir,
        row_ids[neg_consistent_mask],
        row_ids[neg_inconsistent_mask],
        "negative",
    )

    result_df = test_df[["user_id", "item_id", "rating", "review_text"]].copy()
    result_df.insert(0, "test_id", row_ids)
    result_df["sentiment_label"] = labels
    result_df["sentiment_scores"] = [json.dumps(dict(zip(LABEL_ORDER, score_row)), ensure_ascii=False) for score_row in scores]
    result_df["sentiment_pos_status"] = _status_column(row_ids, row_ids[pos_consistent_mask], row_ids[pos_inconsistent_mask])
    result_df["sentiment_neg_status"] = _status_column(row_ids, row_ids[neg_consistent_mask], row_ids[neg_inconsistent_mask])
    result_df.to_csv(os.path.join(output_root, "sentiment_test_predictions.csv"), index=False)

    print(
        "Sentiment preprocessing complete: "
        f"pos consistent={int(pos_consistent_mask.sum())}, "
        f"pos inconsistent={int(pos_inconsistent_mask.sum())}, "
        f"neg consistent={int(neg_consistent_mask.sum())}, "
        f"neg inconsistent={int(neg_inconsistent_mask.sum())}"
    )


def _save_sentiment_view(output_dir, consistent_ids, inconsistent_ids, mode):
    np.save(os.path.join(output_dir, "consistent_id.npy"), consistent_ids.astype(np.int64))
    np.save(os.path.join(output_dir, "inconsistent_id.npy"), inconsistent_ids.astype(np.int64))
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode,
                "id_semantics": "zero-based row index aligned with common/test.csv and model test_*.npy files",
                "num_consistent": int(len(consistent_ids)),
                "num_inconsistent": int(len(inconsistent_ids)),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def _status_column(row_ids, consistent_ids, inconsistent_ids):
    consistent = set(int(idx) for idx in consistent_ids)
    inconsistent = set(int(idx) for idx in inconsistent_ids)
    statuses = []
    for row_id in row_ids:
        row_id_int = int(row_id)
        if row_id_int in consistent:
            statuses.append("consistent")
        elif row_id_int in inconsistent:
            statuses.append("inconsistent")
        else:
            statuses.append("unused")
    return statuses


def map_rating_to_sentiment(rating: float, sentiment_mode: int = 3) -> str:
    """Map a numerical rating to a sentiment category.
    For 5-class mode:
    - 1 -> ``1`` (very negative)
    - 2 -> ``2`` (negative)
    - 3 -> ``3`` (neutral)
    - 4 -> ``4`` (positive)
    - 5 -> ``5`` (very positive)

    Missing or invalid ratings default to neutral (3-class) or "3" (5-class).

    Args:
        rating: Numerical rating value.
        sentiment_mode: Number of sentiment classes (3 or 5).

    Returns:
        Sentiment label string.
    """
    if sentiment_mode == 5:
        if pd.isna(rating):
            return "3"
        try:
            rating_value = float(rating)
        except (TypeError, ValueError):
            return "3"
        return str(int(rating_value))
    
    if pd.isna(rating):
        return "neutral"
    try:
        rating_value = float(rating)
    except (TypeError, ValueError):
        return "neutral"
    if rating_value <= 2.0:
        return "negative"
    if rating_value == 3.0:
        return "neutral"
    return "positive"

def check_consistency(review_sentiment: str, rating_sentiment: str, rating: float, sentiment_mode: int = 3) -> bool:
    """Return whether review sentiment satisfies the rating consistency rule."""
    if sentiment_mode == 5:
        try:
            rating_value = float(rating) if rating is not None and not pd.isna(rating) else None
        except (TypeError, ValueError):
            rating_value = None
        
        if rating_value is None:
            return False
        
        review_val = str(review_sentiment).strip()
        rating_val = str(rating_sentiment).strip()
        
        try:
            review_int = int(review_val)
            rating_int = int(rating_val)
        except (ValueError, TypeError):
            return False
        
        if review_int in [1, 2]:
            return rating_int in [1, 2, 3]
        elif review_int == 3:
            return rating_int == 3
        elif review_int in [4, 5]:
            return rating_int in [4, 5]
        return False
    
    normalized_review = str(review_sentiment).lower()
    normalized_rating = str(rating_sentiment).lower()

    try:
        rating_value = float(rating) if rating is not None and not pd.isna(rating) else None
    except (TypeError, ValueError):
        rating_value = None

    if rating_value == 3.0:
        return normalized_review in {"neutral", "negative"}

    return normalized_review == normalized_rating

def load_sentiment_model(model_name, gpu_id):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def normalize_label(label: str, num_classes: int = 3) -> str:
    normalized = str(label).strip().lower()
    
    if num_classes == 5:
        if "strongly negative" in normalized or normalized.endswith("_0") or normalized == "0":
            return "strongly negative"
        if "negative or neutral" in normalized or normalized.endswith("_2") or normalized == "2":
            return "negative or neutral"
        if "strongly positive" in normalized or normalized.endswith("_4") or normalized == "4":
            return "strongly positive"
        if "negative" in normalized or normalized.endswith("_1") or normalized == "1":
            return "negative"
        if "positive" in normalized or normalized.endswith("_3") or normalized == "3":
            return "positive"
        return "negative or neutral"
    
    if "negative" in normalized or normalized.endswith("_0"):
        return "negative"
    if "neutral" in normalized or normalized.endswith("_1"):
        return "neutral"
    if "positive" in normalized or normalized.endswith("_2"):
        return "positive"
    return "neutral"
    
def predict_sentiments(model_name:str, texts: Sequence[str], batch_size: int = 32, gpu_id: int = 0, num_classes: int = 3) -> tuple[list[str], list[list[float]]]:
    """Predict sentiments for non-empty texts in batches.

    Reviews are tokenized with ``truncation=True`` and ``max_length=512`` to
    satisfy RoBERTa's input length limit.

    Args:
        model_name: Name of the sentiment analysis model.
        texts: List of review texts to analyze.
        batch_size: Number of reviews per batch.
        gpu_id: GPU device ID for inference.
        num_classes: Number of sentiment classes (3 or 5).

    Returns:
        List of sentiment labels.
    """
    tokenizer, model, device = load_sentiment_model(model_name, gpu_id)
    id2label = model.config.id2label
    predictions: list[str] = []
    scores: list[list[float]] = []

    if num_classes == 3:
        target_order = ["negative", "neutral", "positive"]
    else:
        target_order = LABEL_ORDER
    
    normalized_id2label = {
        i: normalize_label(label, num_classes) for i, label in id2label.items()
    }
    label_to_idx = {}
    for i, label in normalized_id2label.items():
        if label not in label_to_idx:
            label_to_idx[label] = i
    
    reorder_idx = []
    for label in target_order:
        if label in label_to_idx:
            reorder_idx.append(label_to_idx[label])
        else:
            reorder_idx.append(0)
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for start in tqdm(
        range(0, len(texts), batch_size), 
        total=total_batches, 
        desc=f"Sentiment analysis ({num_classes}-class)", 
        unit="batch"
    ):

        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu()
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        batch_labels = [normalize_label(id2label[pred_id], num_classes) for pred_id in pred_ids]
        predictions.extend(batch_labels)
        
        batch_scores = probs[:, reorder_idx].cpu().tolist()
        scores.extend([[round(x, 4) for x in row] for row in batch_scores])

    return predictions, scores
