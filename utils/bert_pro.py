import os
import re
import numpy as np
import pandas as pd
import pickle

from collections import Counter, defaultdict
from collections.abc import Sequence
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
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
    output_dir = os.path.join(cfg.data.root, cfg.data.dataset, "bert")
    os.makedirs(output_dir, exist_ok=True)
    print("Starting BERT preprocessing...")
    # Implement BERT-specific preprocessing steps here
    train_df = train_df.copy().reset_index(drop=True)
    valid_df = valid_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)
    train_df["review_idx"] = np.arange(len(train_df), dtype=np.int64)

    train_df["clean_review"] = bert_clean_review(train_df)

     # 2. train review embedding 추출
    texts = train_df["clean_review"].fillna("").astype(str).tolist()

    vec_dim = 128
    batch_size = 16
    gpu_id = int(cfg.experiment.get("device", 0))

    review_emb = get_bert_whitening_embeddings(
        texts,
        vec_dim=vec_dim,
        batch_size=batch_size,
        gpu_id=gpu_id,
    )
    review_emb = np.asarray(review_emb, dtype=np.float32)

        # 3. 전체 user/item 개수 계산
    all_df = pd.concat([train_df, valid_df, test_df], axis=0)
    num_users = int(all_df["user_id"].max()) + 1
    num_items = int(all_df["item_id"].max()) + 1
    del all_df

    user_ids = train_df["user_id"].astype(np.int64).to_numpy()
    item_ids = train_df["item_id"].astype(np.int64).to_numpy()
    ratings = train_df["rating"].astype(np.float32).to_numpy()

    # 4. user/item 전체 review embedding 평균
    user_review_emb = aggregate_embedding_by_id(ids=user_ids, emb=review_emb, num_entities=num_users)
    item_review_emb = aggregate_embedding_by_id(ids=item_ids, emb=review_emb, num_entities=num_items)

    # 5. user like/dislike embedding
    # 기본값: rating >= 4.0 을 like로 둠
    like_threshold = float(4.0)
    like_mask = ratings >= like_threshold
    dislike_mask = ~like_mask

    user_like_emb = aggregate_embedding_by_id(ids=user_ids[like_mask], emb=review_emb[like_mask], num_entities=num_users)
    user_dislike_emb = aggregate_embedding_by_id(ids=user_ids[dislike_mask], emb=review_emb[dislike_mask], num_entities=num_users)

    # 6. RGCL / RecAFR / LETTER용 random split 고정
    seed = 64
    rng = np.random.default_rng(seed)

    user_review_emb_s1, user_review_emb_s2, user_review_split_idx = build_random_split_entity_embeddings(df=train_df,emb=review_emb,lead_col="user_id",num_entities=num_users,rng=rng)
    item_review_emb_s1, item_review_emb_s2, item_review_split_idx = build_random_split_entity_embeddings(df=train_df,emb=review_emb,lead_col="item_id",num_entities=num_items,rng=rng)

    # 7. 저장
    np.save(os.path.join(output_dir, "review_emb.npy"), review_emb)

    np.save(os.path.join(output_dir, "user_review_emb.npy"), user_review_emb)
    np.save(os.path.join(output_dir, "item_review_emb.npy"), item_review_emb)

    np.save(os.path.join(output_dir, "user_like_emb.npy"), user_like_emb)
    np.save(os.path.join(output_dir, "user_dislike_emb.npy"), user_dislike_emb)

    np.save(os.path.join(output_dir, "user_review_emb_s1.npy"), user_review_emb_s1)
    np.save(os.path.join(output_dir, "user_review_emb_s2.npy"), user_review_emb_s2)
    np.save(os.path.join(output_dir, "item_review_emb_s1.npy"), item_review_emb_s1)
    np.save(os.path.join(output_dir, "item_review_emb_s2.npy"), item_review_emb_s2)

    with open(os.path.join(output_dir, "user_review_split_idx.pkl"), "wb") as f:
        pickle.dump(user_review_split_idx, f)

    with open(os.path.join(output_dir, "item_review_split_idx.pkl"), "wb") as f:
        pickle.dump(item_review_split_idx, f)

    # split별 id/rating만 저장. valid/test review는 안 씀.
    save_bert_split_ids("train", train_df, output_dir)
    save_bert_split_ids("valid", valid_df, output_dir)
    save_bert_split_ids("test", test_df, output_dir)


def aggregate_embedding_by_id(ids, emb, num_entities):
    """
    ids: [N]
    emb: [N, dim]

    return:
        [num_entities, dim]
    """
    dim = emb.shape[1]

    out = np.zeros((num_entities, dim), dtype=np.float32)
    cnt = np.zeros(num_entities, dtype=np.int64)

    for idx, entity_id in enumerate(ids):
        entity_id = int(entity_id)
        out[entity_id] += emb[idx]
        cnt[entity_id] += 1

    nonzero = cnt > 0
    out[nonzero] = out[nonzero] / cnt[nonzero, None]

    return out


def build_random_split_entity_embeddings(df, emb, lead_col, num_entities, rng):
    """
    각 user 또는 item의 train reviews를 한 번 random split해서
    S^1, S^2 embedding 평균을 만든다.

    return:
        emb_s1: [num_entities, dim]
        emb_s2: [num_entities, dim]
        split_idx: dict
            split_idx[entity_id] = {
                "s1": [train row indices],
                "s2": [train row indices],
            }
    """
    dim = emb.shape[1]

    emb_s1 = np.zeros((num_entities, dim), dtype=np.float32)
    emb_s2 = np.zeros((num_entities, dim), dtype=np.float32)

    split_idx = {}

    grouped = df.groupby(lead_col)["review_idx"].apply(list).to_dict()

    for entity_id, idx_list in grouped.items():
        entity_id = int(entity_id)
        idx_arr = np.asarray(idx_list, dtype=np.int64)

        if len(idx_arr) == 0:
            s1_idx = np.asarray([], dtype=np.int64)
            s2_idx = np.asarray([], dtype=np.int64)

        elif len(idx_arr) == 1:
            # review가 1개뿐이면 둘 다 같은 review를 사용.
            # 완전 partition으로 두면 한쪽이 zero가 되므로 contrastive view가 불안정해짐.
            s1_idx = idx_arr
            s2_idx = idx_arr

        else:
            shuffled = rng.permutation(idx_arr)
            cut = len(shuffled) // 2

            # 양쪽이 비지 않도록 보장
            cut = max(1, min(cut, len(shuffled) - 1))

            s1_idx = shuffled[:cut]
            s2_idx = shuffled[cut:]

        if len(s1_idx) > 0:
            emb_s1[entity_id] = emb[s1_idx].mean(axis=0)

        if len(s2_idx) > 0:
            emb_s2[entity_id] = emb[s2_idx].mean(axis=0)

        split_idx[entity_id] = {
            "s1": s1_idx.tolist(),
            "s2": s2_idx.tolist(),
        }

    return emb_s1, emb_s2, split_idx


def save_bert_split_ids(split_name, df, output_dir):
    user_ids = df["user_id"].astype(np.int64).to_numpy()
    item_ids = df["item_id"].astype(np.int64).to_numpy()
    ratings = df["rating"].astype(np.float32).to_numpy()

    np.save(os.path.join(output_dir, f"{split_name}_user_id.npy"), user_ids)
    np.save(os.path.join(output_dir, f"{split_name}_item_id.npy"), item_ids)
    np.save(os.path.join(output_dir, f"{split_name}_rating.npy"), ratings)

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

def compute_kernel_bias(vecs, vec_dim):
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, _ = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :vec_dim], -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


def get_bert_whitening_embeddings(texts: Sequence[str], vec_dim: int = 128, batch_size: int = 32, gpu_id: int = 0):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    loaded_model = BertModel.from_pretrained("bert-base-uncased")
    if not isinstance(loaded_model, BertModel):
        raise TypeError("bert-base-uncased did not load as a BertModel")
    model = loaded_model

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    torch.nn.Module.to(model, device)
    model.eval()
    model.config.output_hidden_states = True

    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for start in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="BERT-Whitening", unit="batch"):
        batch_texts = list(texts[start:start + batch_size])
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        output1 = outputs.hidden_states[-2]
        output2 = outputs.hidden_states[-1]
        last2 = (output1 + output2) / 2
        last2 = torch.sum(attention_mask.unsqueeze(-1) * last2, dim=1) / attention_mask.sum(dim=1, keepdims=True)
        embeddings.append(last2.cpu().numpy())

    if not embeddings:
        return []
    vecs = np.vstack(embeddings)
    kernel, bias = compute_kernel_bias(vecs, vec_dim)
    vecs = transform_and_normalize(vecs, kernel, bias)

    return vecs.tolist()
