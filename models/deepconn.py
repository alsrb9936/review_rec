import numpy as np
import torch
import torch.nn as nn


class DeepCoNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        embedding_matrix = np.load(cfg.data.word_embedding_path)
        embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

        self.word_embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=bool(cfg.model.freeze_word_embedding),
            padding_idx=int(cfg.data.pad_id),
        )

        embedding_dim = embedding_matrix.shape[1]

        # 임시 예시: 실제 DeepCoNN CNN encoder는 여기에 구현
        self.user_proj = nn.Linear(embedding_dim, cfg.model.hidden_dim)
        self.item_proj = nn.Linear(embedding_dim, cfg.model.hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(cfg.model.hidden_dim * 2, cfg.model.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.model.hidden_dim, 1),
        )

    def forward(self, batch):
        user_reviews = batch["user_reviews"]
        item_reviews = batch["item_reviews"]

        user_emb = self.word_embedding(user_reviews)
        item_emb = self.word_embedding(item_reviews)

        # user_emb: (B, review_count, review_length, embedding_dim)
        # 여기서는 단순 평균. 나중에 CNN으로 교체하면 됨.
        user_vec = user_emb.mean(dim=(1, 2))
        item_vec = item_emb.mean(dim=(1, 2))

        user_vec = self.user_proj(user_vec)
        item_vec = self.item_proj(item_vec)

        out = self.predictor(torch.cat([user_vec, item_vec], dim=-1))
        return out