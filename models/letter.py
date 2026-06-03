import os

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


class LETTER(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)
        self.hidden_dim = int(cfg.model.hidden_dim)
        self.edge_ratio = float(cfg.model.edge_ratio)

        data_dir = os.path.join(cfg.data.root, cfg.data.dataset, cfg.data.type)
        user_review = self._load_embedding(data_dir, "user_review_emb.npy", self.num_users)
        item_review = self._load_embedding(data_dir, "item_review_emb.npy", self.num_items)
        user_like = self._load_embedding(data_dir, "user_like_emb.npy", self.num_users)
        user_dislike = self._load_embedding(data_dir, "user_dislike_emb.npy", self.num_users)

        embedding_dim = int(user_review.shape[1])
        configured_dim = int(cfg.model.get("embedding_dim", embedding_dim))
        if configured_dim != embedding_dim:
            raise ValueError(
                f"LETTER embedding_dim={configured_dim} does not match user_review_emb.npy dim={embedding_dim}."
            )
        for name, embedding in {
            "item_review_emb.npy": item_review,
            "user_like_emb.npy": user_like,
            "user_dislike_emb.npy": user_dislike,
        }.items():
            if int(embedding.shape[1]) != embedding_dim:
                raise ValueError(f"{name} dim={embedding.shape[1]} does not match {embedding_dim}.")

        self.user_embedding = nn.Embedding.from_pretrained(torch.from_numpy(user_review), freeze=True)
        self.item_embedding = nn.Embedding.from_pretrained(torch.from_numpy(item_review), freeze=True)
        self.user_pos_embedding = nn.Embedding.from_pretrained(torch.from_numpy(user_like), freeze=True)
        self.user_neg_embedding = nn.Embedding.from_pretrained(torch.from_numpy(user_dislike), freeze=True)

        # The provided BERT artifacts contain user like/dislike embeddings and a single item embedding.
        # Reuse the item review embedding for positive/negative item branches to preserve the original
        # LETTER architecture without inventing unavailable item sentiment features.
        self.item_pos_embedding = nn.Embedding.from_pretrained(torch.from_numpy(item_review.copy()), freeze=True)
        self.item_neg_embedding = nn.Embedding.from_pretrained(torch.from_numpy(item_review.copy()), freeze=True)

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.user_p = nn.Embedding(self.num_users, self.hidden_dim)
        self.item_p = nn.Embedding(self.num_items, self.hidden_dim)
        nn.init.xavier_uniform_(self.user_p.weight)
        nn.init.xavier_uniform_(self.item_p.weight)

        dropout = float(cfg.model.get("dropout", 0.0))
        self.ruFC = self._projection_block(embedding_dim, self.hidden_dim, dropout)
        self.riFC = self._projection_block(embedding_dim, self.hidden_dim, dropout)
        self.rupFC = self._projection_block(embedding_dim, self.hidden_dim, dropout)
        self.runFC = self._projection_block(embedding_dim, self.hidden_dim, dropout)
        self.ripFC = self._projection_block(embedding_dim, self.hidden_dim, dropout)
        self.rinFC = self._projection_block(embedding_dim, self.hidden_dim, dropout)

        self.ugnn1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ignn1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.upnn1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ipnn1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.unnn1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.innn1 = nn.Linear(self.hidden_dim, self.hidden_dim)

        user_rating, item_rating = self._load_ratings(data_dir)
        self.register_buffer("user_rating", torch.from_numpy(user_rating.astype(np.float32)))
        self.register_buffer("item_rating", torch.from_numpy(item_rating.astype(np.float32)))
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def _projection_block(input_dim: int, hidden_dim: int, dropout: float):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _load_embedding(data_dir: str, filename: str, expected_rows: int):
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing LETTER embedding file: {path}")
        embedding = np.load(path).astype(np.float32)
        if embedding.ndim != 2:
            raise ValueError(f"{filename} must be a 2D array, got shape={embedding.shape}.")
        if int(embedding.shape[0]) < expected_rows:
            raise ValueError(
                f"{filename} has {embedding.shape[0]} rows but stats require at least {expected_rows}."
            )
        return embedding

    def _load_ratings(self, data_dir: str):
        user_ids = np.load(os.path.join(data_dir, "train_user_id.npy")).astype(np.int64)
        item_ids = np.load(os.path.join(data_dir, "train_item_id.npy")).astype(np.int64)
        ratings = np.load(os.path.join(data_dir, "train_rating.npy")).astype(np.float32)
        if not (len(user_ids) == len(item_ids) == len(ratings)):
            raise ValueError(
                "LETTER train arrays must align: "
                f"users={len(user_ids)}, items={len(item_ids)}, ratings={len(ratings)}"
            )

        user_rating = np.zeros((self.num_users, self.num_items), dtype=np.float32)
        item_rating = np.zeros((self.num_items, self.num_users), dtype=np.float32)
        user_rating[user_ids, item_ids] = ratings
        item_rating[item_ids, user_ids] = ratings
        return user_rating, item_rating

    def graph_edge__(self, user_emb, item_emb, user_ids, item_ids):
        ratio = max(0.0, min(100.0, self.edge_ratio)) / 100.0
        norm_u = torch.nn.functional.normalize(user_emb, p=2, dim=1)
        norm_i = torch.nn.functional.normalize(item_emb, p=2, dim=1)

        num_u = max(1, min(self.num_users, int(np.ceil(ratio * self.num_users))))
        num_i = max(1, min(self.num_items, int(np.ceil(ratio * self.num_items))))

        def calculate_mask_in_batches(norm_matrix, num_elements, ids, top_k):
            similarities = torch.mm(norm_matrix.index_select(0, ids), norm_matrix.t())
            probs = torch.softmax(similarities, dim=1)
            selected_indices = torch.multinomial(probs, top_k, replacement=False)

            mask = torch.ones((ids.size(0), num_elements), dtype=torch.bool, device=norm_matrix.device)
            mask[torch.arange(ids.size(0), device=norm_matrix.device).unsqueeze(1), selected_indices] = False
            selected_logits = similarities.masked_fill(mask, torch.finfo(similarities.dtype).min)
            selected_probs = torch.softmax(selected_logits, dim=1)
            return mask, selected_probs

        u_mask, u_sims = calculate_mask_in_batches(norm_u, self.num_users, user_ids, num_u)
        i_mask, i_sims = calculate_mask_in_batches(norm_i, self.num_items, item_ids, num_i)
        return u_mask, i_mask, u_sims, i_sims

    def forward(self, user_ids, item_ids, rating=None, clip=False):
        u_mask, i_mask, u_sims, i_sims = self.graph_edge__(
            self.user_rating / 5.0,
            self.item_rating / 5.0,
            user_ids,
            item_ids,
        )

        unique_u_mask = (~u_mask).any(dim=0)
        unique_i_mask = (~i_mask).any(dim=0)
        unique_u_mask[user_ids] = True
        unique_i_mask[item_ids] = True

        unique_user_ids = torch.nonzero(unique_u_mask, as_tuple=False).squeeze(1)
        unique_item_ids = torch.nonzero(unique_i_mask, as_tuple=False).squeeze(1)
        t_user_ids = torch.searchsorted(unique_user_ids, user_ids)
        t_item_ids = torch.searchsorted(unique_item_ids, item_ids)

        user_embeds = self.ruFC(self.user_embedding.weight[unique_u_mask])
        item_embeds = self.riFC(self.item_embedding.weight[unique_i_mask])
        user_pos_embeds = self.rupFC(self.user_pos_embedding.weight[unique_u_mask])
        user_neg_embeds = self.runFC(self.user_neg_embedding.weight[unique_u_mask])
        item_pos_embeds = self.ripFC(self.item_pos_embedding.weight[unique_i_mask])
        item_neg_embeds = self.rinFC(self.item_neg_embedding.weight[unique_i_mask])

        user_r_embeds = self.ugnn1(torch.mm(u_sims[:, unique_u_mask], user_embeds))
        item_r_embeds = self.ignn1(torch.mm(i_sims[:, unique_i_mask], item_embeds))
        user_r_embeds = user_r_embeds + user_embeds[t_user_ids]
        item_r_embeds = item_r_embeds + item_embeds[t_item_ids]

        user_pos_r_embeds = self.upnn1(torch.mm(u_sims[:, unique_u_mask], user_pos_embeds))
        user_pos_r_embeds = user_pos_r_embeds + user_pos_embeds[t_user_ids]

        item_pos_r_embeds = self.ipnn1(torch.mm(i_sims[:, unique_i_mask], item_pos_embeds))
        item_pos_r_embeds = item_pos_r_embeds + item_pos_embeds[t_item_ids]

        user_neg_r_embeds = self.unnn1(torch.mm(u_sims[:, unique_u_mask], user_neg_embeds))
        user_neg_r_embeds = user_neg_r_embeds + user_neg_embeds[t_user_ids]

        item_neg_r_embeds = self.innn1(torch.mm(i_sims[:, unique_i_mask], item_neg_embeds))
        item_neg_r_embeds = item_neg_r_embeds + item_neg_embeds[t_item_ids]

        user_biases = self.user_bias(user_ids).squeeze(-1)
        item_biases = self.item_bias(item_ids).squeeze(-1)
        dot_product = (user_r_embeds * item_r_embeds).sum(1)
        dot_product = dot_product + (user_pos_r_embeds * item_r_embeds).sum(1)
        dot_product = dot_product - (user_neg_r_embeds * item_r_embeds).sum(1)

        prediction = dot_product + user_biases + item_biases
        if clip:
            prediction = torch.clamp(prediction, 1.0, 5.0)
        return prediction

    def calculate_loss(self, user_id, item_id, rating):
        prediction = self.forward(user_id, item_id, rating, clip=False)
        return self.loss_fn(prediction, rating)
