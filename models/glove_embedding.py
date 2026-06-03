import os

import numpy as np
import torch
import torch.nn as nn


def build_glove_embedding(cfg) -> nn.Embedding:
    data_dir = os.path.join(cfg.data.root, cfg.data.dataset, cfg.data.type)
    word_emb_path = os.path.join(data_dir, "word_emb.npy")
    if not os.path.exists(word_emb_path):
        raise FileNotFoundError(
            f"Missing word_emb.npy: {word_emb_path}. Run GloVe preprocessing again."
        )

    word_emb = np.load(word_emb_path).astype(np.float32)
    weight = torch.tensor(word_emb, dtype=torch.float32)
    freeze = bool(cfg.model.get("freeze_word_embedding", True))
    return nn.Embedding.from_pretrained(weight, freeze=freeze, padding_idx=0)
