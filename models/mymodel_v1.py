# models/mymodel.py

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def _get_bert_data_dir(cfg: DictConfig) -> str:
    data_type = str(cfg.data.get("type", "bert"))
    if data_type.lower() in {"none", "null", ""}:
        data_type = "bert"
    return os.path.join(cfg.data.root, cfg.data.dataset, data_type)


def _build_norm_adj(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    num_nodes = num_users + num_items
    item_nodes = item_ids + num_users

    rows = np.concatenate([user_ids, item_nodes])
    cols = np.concatenate([item_nodes, user_ids])

    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

    adj = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(num_nodes, num_nodes),
    ).coalesce()

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    row, col = adj.indices()
    values = adj.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return torch.sparse_coo_tensor(
        adj.indices(),
        values,
        size=adj.shape,
    ).coalesce()


class MyModelV1(nn.Module):
    """
    Review-free inference model.

    Main prediction:
        CF-only rating prediction.

    Training auxiliary:
        train review edge embeddings -> user/item review graph teacher
        CF embeddings are aligned to review graph teacher embeddings.

    Inference:
        review graph is not used by forward().
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)
        self.embedding_dim = int(cfg.model.embedding_dim)
        self.num_layers = int(cfg.model.get("num_layers", 2))
        self.review_layers = int(cfg.model.get("review_layers", 1))
        self.temperature = float(cfg.model.get("temperature", 0.2))
        self.lambda_user_align = float(cfg.model.get("lambda_user_align", 0.05))
        self.lambda_item_align = float(cfg.model.get("lambda_item_align", 0.05))

        data_dir = _get_bert_data_dir(cfg)

        train_user = np.load(os.path.join(data_dir, "train_user_id.npy")).astype(np.int64)
        train_item = np.load(os.path.join(data_dir, "train_item_id.npy")).astype(np.int64)
        train_rating = np.load(os.path.join(data_dir, "train_rating.npy")).astype(np.float32)
        review_emb = np.load(os.path.join(data_dir, "review_emb.npy")).astype(np.float32)

        if not (len(train_user) == len(train_item) == len(train_rating) == len(review_emb)):
            raise ValueError(
                "train_user_id, train_item_id, train_rating, review_emb must align. "
                f"users={len(train_user)}, items={len(train_item)}, "
                f"ratings={len(train_rating)}, reviews={len(review_emb)}"
            )

        self.review_dim = int(review_emb.shape[1])

        norm_adj = _build_norm_adj(
            user_ids=train_user,
            item_ids=train_item,
            num_users=self.num_users,
            num_items=self.num_items,
        )

        self.register_buffer("norm_adj", norm_adj, persistent=False)
        self.register_buffer("train_user", torch.from_numpy(train_user).long(), persistent=False)
        self.register_buffer("train_item", torch.from_numpy(train_item).long(), persistent=False)
        self.register_buffer("train_rating", torch.from_numpy(train_rating).float(), persistent=False)
        self.register_buffer("train_review_emb", torch.from_numpy(review_emb).float(), persistent=False)

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Review-side encoders.
        # Same review is viewed differently from user-side and item-side.
        self.review_norm = nn.LayerNorm(self.review_dim)
        self.user_review_proj = nn.Sequential(
            nn.Linear(self.review_dim + 1, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(float(cfg.model.get("review_dropout", 0.1))),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.item_review_proj = nn.Sequential(
            nn.Linear(self.review_dim + 1, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(float(cfg.model.get("review_dropout", 0.1))),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        # Projection heads for contrastive alignment.
        self.cf_user_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.cf_item_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.text_user_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.text_item_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.loss_fn = nn.MSELoss()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        for module in [
            self.user_review_proj,
            self.item_review_proj,
            self.cf_user_proj,
            self.cf_item_proj,
            self.text_user_proj,
            self.text_item_proj,
        ]:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def get_cf_embeddings(self):
        emb = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0,
        )
        outputs = [emb]

        for _ in range(self.num_layers):
            emb = torch.sparse.mm(self.norm_adj, emb)
            outputs.append(emb)

        final = torch.stack(outputs, dim=0).mean(dim=0)
        user_cf, item_cf = torch.split(final, [self.num_users, self.num_items], dim=0)
        return user_cf, item_cf

    def _aggregate_edge_reviews(self):
        """
        One-hop edge-review aggregation:
            review edge -> user node
            review edge -> item node
        """
        review = self.review_norm(self.train_review_emb)

        # Normalize rating roughly to [0, 1].
        rating_feat = (self.train_rating.view(-1, 1) / 5.0).clamp(0.0, 1.0)
        review_input = torch.cat([review, rating_feat], dim=-1)

        user_msg = self.user_review_proj(review_input)
        item_msg = self.item_review_proj(review_input)

        user_text = torch.zeros(
            self.num_users,
            self.embedding_dim,
            device=review.device,
            dtype=review.dtype,
        )
        item_text = torch.zeros(
            self.num_items,
            self.embedding_dim,
            device=review.device,
            dtype=review.dtype,
        )

        user_text.index_add_(0, self.train_user, user_msg)
        item_text.index_add_(0, self.train_item, item_msg)

        user_deg = torch.bincount(self.train_user, minlength=self.num_users).float().to(review.device)
        item_deg = torch.bincount(self.train_item, minlength=self.num_items).float().to(review.device)

        user_text = user_text / user_deg.clamp_min(1.0).unsqueeze(-1)
        item_text = item_text / item_deg.clamp_min(1.0).unsqueeze(-1)

        return user_text, item_text

    def get_review_graph_embeddings(self):
        """
        Review graph teacher.

        Initial node features come from review-edge aggregation.
        Then we propagate them on the same train user-item graph.
        """
        user_text, item_text = self._aggregate_edge_reviews()
        h = torch.cat([user_text, item_text], dim=0)

        outputs = [h]
        for _ in range(self.review_layers):
            h = torch.sparse.mm(self.norm_adj, h)
            outputs.append(h)

        final = torch.stack(outputs, dim=0).mean(dim=0)
        user_review, item_review = torch.split(final, [self.num_users, self.num_items], dim=0)
        return user_review, item_review

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor):
        """
        Review-free inference path.
        """
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)

        user_cf_all, item_cf_all = self.get_cf_embeddings()

        user_cf = user_cf_all[user_id]
        item_cf = item_cf_all[item_id]

        pred = torch.sum(user_cf * item_cf, dim=-1)
        pred = pred + self.user_bias(user_id).squeeze(-1)
        pred = pred + self.item_bias(item_id).squeeze(-1)
        pred = pred + self.global_bias

        return pred

    def _info_nce(self, query: torch.Tensor, key: torch.Tensor):
        """
        In-batch InfoNCE.
        query: [B, D]
        key:   [B, D]
        """
        if query.shape[0] <= 1:
            return query.new_tensor(0.0)

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        logits = query @ key.t()
        logits = logits / self.temperature

        labels = torch.arange(query.shape[0], device=query.device)
        return F.cross_entropy(logits, labels)

    def alignment_loss(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_cf_all: torch.Tensor,
        item_cf_all: torch.Tensor,
    ):
        user_review_all, item_review_all = self.get_review_graph_embeddings()

        unique_users = torch.unique(user_id.view(-1))
        unique_items = torch.unique(item_id.view(-1))

        user_q = self.cf_user_proj(user_cf_all[unique_users])

        # Stop-gradient on review graph representation, not on text projector.
        # This keeps raw review teacher stable while still training text projection.
        user_k = self.text_user_proj(user_review_all[unique_users].detach())

        item_q = self.cf_item_proj(item_cf_all[unique_items])
        item_k = self.text_item_proj(item_review_all[unique_items].detach())

        loss_user = self._info_nce(user_q, user_k)
        loss_item = self._info_nce(item_q, item_k)

        return loss_user, loss_item

    def calculate_loss(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        rating: torch.Tensor,
        return_dict: bool = False,
    ):
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)
        rating = rating.view(-1).float()

        user_cf_all, item_cf_all = self.get_cf_embeddings()

        user_cf = user_cf_all[user_id]
        item_cf = item_cf_all[item_id]

        pred = torch.sum(user_cf * item_cf, dim=-1)
        pred = pred + self.user_bias(user_id).squeeze(-1)
        pred = pred + self.item_bias(item_id).squeeze(-1)
        pred = pred + self.global_bias

        rating_loss = self.loss_fn(pred, rating)

        user_align_loss, item_align_loss = self.alignment_loss(
            user_id=user_id,
            item_id=item_id,
            user_cf_all=user_cf_all,
            item_cf_all=item_cf_all,
        )

        total_loss = (
            rating_loss
            + self.lambda_user_align * user_align_loss
            + self.lambda_item_align * item_align_loss
        )

        if return_dict:
            return {
                "loss": total_loss,
                "rating_loss": rating_loss.detach(),
                "user_align_loss": user_align_loss.detach(),
                "item_align_loss": item_align_loss.detach(),
            }

        return total_loss