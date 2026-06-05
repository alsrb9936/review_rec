# models/mymodel_v2.py

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class RatingGraphEncoder(nn.Module):
    """
    LightGCN-style rating graph encoder.

    This class owns user/item ID embeddings and performs graph propagation.
    MyModelV2 must use this instead of defining separate user/item embeddings.
    """

    def __init__(self, num_users, num_items, norm_adj, d_id=64, num_layers=2):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.d_id = int(d_id)
        self.num_layers = int(num_layers)

        self.user_embedding = nn.Embedding(self.num_users, self.d_id)
        self.item_embedding = nn.Embedding(self.num_items, self.d_id)
        self.register_buffer("norm_adj", norm_adj.coalesce(), persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_all_embeddings(self):
        x = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0,
        )
        outs = [x]

        for _ in range(self.num_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            outs.append(x)

        all_emb = torch.stack(outs, dim=0).mean(dim=0)

        user_all = all_emb[: self.num_users]
        item_all = all_emb[self.num_users :]

        return user_all, item_all

    def forward(self, user_ids, item_ids):
        user_all, item_all = self.get_all_embeddings()
        return user_all[user_ids], item_all[item_ids]


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

    edge_index = torch.tensor(
        np.stack([rows, cols], axis=0),
        dtype=torch.long,
    )
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

    norm_adj = torch.sparse_coo_tensor(
        adj.indices(),
        values,
        size=adj.shape,
    ).coalesce()

    return norm_adj


class MyModelV2(nn.Module):
    """
    Pair-only review-free inference model.

    Prediction:
        CF-only rating prediction.

    Training auxiliary:
        train review edge embeddings
        -> user/item review graph embeddings
        -> review pair embedding
        -> CF pair-guided orthogonal decomposition
        -> pair-only contrastive alignment

    Removed:
        user_align_loss
        item_align_loss
        node_alignment_loss

    Used:
        pair_align_loss only.

    Pair decomposition:
        cf_pair     = [user_cf || item_cf]
        review_pair = [user_review || item_review]

        review_parallel = projection of review_pair onto cf_pair direction
        review_residual = review_pair - review_parallel

        review_used = review_parallel + residual_weight * review_residual

    Default:
        residual_weight = 0.0
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
        self.lambda_pair_align = float(cfg.model.get("lambda_pair_align", 0.05))

        self.orthogonal_residual_weight = float(
            cfg.model.get("orthogonal_residual_weight", 0.0)
        )
        self.orthogonal_eps = float(cfg.model.get("orthogonal_eps", 1e-8))
        self.rating_scale = float(cfg.model.get("rating_scale", 5.0))

        data_dir = _get_bert_data_dir(cfg)

        train_user = np.load(
            os.path.join(data_dir, "train_user_id.npy")
        ).astype(np.int64)
        train_item = np.load(
            os.path.join(data_dir, "train_item_id.npy")
        ).astype(np.int64)
        train_rating = np.load(
            os.path.join(data_dir, "train_rating.npy")
        ).astype(np.float32)
        review_emb = np.load(
            os.path.join(data_dir, "review_emb.npy")
        ).astype(np.float32)

        if not (
            len(train_user)
            == len(train_item)
            == len(train_rating)
            == len(review_emb)
        ):
            raise ValueError(
                "train_user_id.npy, train_item_id.npy, train_rating.npy, "
                "review_emb.npy must have the same first dimension. "
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

        # CF graph encoder is the only owner of user/item ID embeddings.
        self.graph_encoder = RatingGraphEncoder(
            num_users=self.num_users,
            num_items=self.num_items,
            norm_adj=norm_adj,
            d_id=self.embedding_dim,
            num_layers=self.num_layers,
        )

        # Train-only review graph buffers.
        self.register_buffer(
            "train_user",
            torch.from_numpy(train_user).long(),
            persistent=False,
        )
        self.register_buffer(
            "train_item",
            torch.from_numpy(train_item).long(),
            persistent=False,
        )
        self.register_buffer(
            "train_rating",
            torch.from_numpy(train_rating).float(),
            persistent=False,
        )
        self.register_buffer(
            "train_review_emb",
            torch.from_numpy(review_emb).float(),
            persistent=False,
        )

        # Rating bias terms.
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Review-side edge encoders.
        self.review_norm = nn.LayerNorm(self.review_dim)
        review_dropout = float(cfg.model.get("review_dropout", 0.1))

        self.user_review_encoder = nn.Sequential(
            nn.Linear(self.review_dim + 1, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(review_dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        self.item_review_encoder = nn.Sequential(
            nn.Linear(self.review_dim + 1, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(review_dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        # Pair-level projection heads.
        pair_dim = self.embedding_dim * 2
        self.cf_pair_proj = nn.Linear(pair_dim, pair_dim, bias=False)
        self.review_pair_proj = nn.Linear(pair_dim, pair_dim, bias=False)

        self.loss_fn = nn.MSELoss()

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.global_bias)

        modules = [
            self.user_review_encoder,
            self.item_review_encoder,
            self.cf_pair_proj,
            self.review_pair_proj,
        ]

        for module in modules:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def _aggregate_edge_reviews(self):
        """
        Edge review -> user/item node aggregation.

        Each train review is first treated as an edge feature.
        The same review is projected into:
            user-side review message
            item-side review message
        """
        review = self.review_norm(self.train_review_emb)

        rating_feat = (
            self.train_rating.view(-1, 1) / self.rating_scale
        ).clamp(0.0, 1.0)

        review_input = torch.cat([review, rating_feat], dim=-1)

        user_msg = self.user_review_encoder(review_input)
        item_msg = self.item_review_encoder(review_input)

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

        user_deg = torch.bincount(
            self.train_user,
            minlength=self.num_users,
        ).float().to(review.device)

        item_deg = torch.bincount(
            self.train_item,
            minlength=self.num_items,
        ).float().to(review.device)

        user_text = user_text / user_deg.clamp_min(1.0).unsqueeze(-1)
        item_text = item_text / item_deg.clamp_min(1.0).unsqueeze(-1)

        return user_text, item_text

    def get_review_graph_embeddings(self):
        """
        Review graph teacher.

        Initial user/item review features come from edge-review aggregation.
        Then they are propagated over the same train user-item graph used by
        RatingGraphEncoder.
        """
        user_text, item_text = self._aggregate_edge_reviews()
        h = torch.cat([user_text, item_text], dim=0)

        outputs = [h]

        # Reuse RatingGraphEncoder's normalized adjacency.
        norm_adj = self.graph_encoder.norm_adj

        for _ in range(self.review_layers):
            h = torch.sparse.mm(norm_adj, h)
            outputs.append(h)

        final = torch.stack(outputs, dim=0).mean(dim=0)

        user_review, item_review = torch.split(
            final,
            [self.num_users, self.num_items],
            dim=0,
        )

        return user_review, item_review

    def predict_from_cf(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_cf_all: torch.Tensor,
        item_cf_all: torch.Tensor,
    ):
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)

        user_cf = user_cf_all[user_id]
        item_cf = item_cf_all[item_id]

        pred = torch.sum(user_cf * item_cf, dim=-1)

        pred = pred + self.user_bias(user_id).squeeze(-1)
        pred = pred + self.item_bias(item_id).squeeze(-1)
        pred = pred + self.global_bias

        return pred

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor):
        """
        Review-free inference path.
        """
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)

        user_cf_all, item_cf_all = self.graph_encoder.get_all_embeddings()

        return self.predict_from_cf(
            user_id=user_id,
            item_id=item_id,
            user_cf_all=user_cf_all,
            item_cf_all=item_cf_all,
        )

    def _info_nce(self, query: torch.Tensor, key: torch.Tensor):
        """
        In-batch InfoNCE.
        """
        if query.shape[0] <= 1:
            return query.new_tensor(0.0)

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        logits = query @ key.t()
        logits = logits / self.temperature

        labels = torch.arange(query.shape[0], device=query.device)

        return F.cross_entropy(logits, labels)

    def _decompose_review_by_cf(
        self,
        review_pair: torch.Tensor,
        cf_pair: torch.Tensor,
    ):
        """
        Orthogonal decomposition of review pair embedding by CF pair direction.

        review_pair:
            [B, 2D] = [user_review || item_review]

        cf_pair:
            [B, 2D] = [user_cf || item_cf]
        """
        basis = F.normalize(
            cf_pair.detach(),
            dim=-1,
            eps=self.orthogonal_eps,
        )

        scalar = torch.sum(review_pair * basis, dim=-1, keepdim=True)
        review_parallel = scalar * basis
        review_residual = review_pair - review_parallel

        return review_parallel, review_residual

    def pair_orthogonal_alignment_loss(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_cf_all: torch.Tensor,
        item_cf_all: torch.Tensor,
        user_review_all: torch.Tensor,
        item_review_all: torch.Tensor,
    ):
        """
        Pair-only orthogonal CF-review alignment.

        No user_align_loss.
        No item_align_loss.
        """
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)

        user_cf = user_cf_all[user_id]
        item_cf = item_cf_all[item_id]

        user_review = user_review_all[user_id]
        item_review = item_review_all[item_id]

        cf_pair_raw = torch.cat([user_cf, item_cf], dim=-1)
        review_pair_raw = torch.cat([user_review, item_review], dim=-1)

        review_parallel, review_residual = self._decompose_review_by_cf(
            review_pair=review_pair_raw,
            cf_pair=cf_pair_raw,
        )

        review_used = (
            review_parallel
            + self.orthogonal_residual_weight * review_residual
        )

        cf_pair_q = self.cf_pair_proj(cf_pair_raw)
        review_pair_k = self.review_pair_proj(review_used)

        pair_loss = self._info_nce(cf_pair_q, review_pair_k)

        residual_cos = F.cosine_similarity(
            review_residual,
            cf_pair_raw.detach(),
            dim=-1,
            eps=self.orthogonal_eps,
        ).abs().mean()

        parallel_norm = review_parallel.norm(dim=-1).mean()
        residual_norm = review_residual.norm(dim=-1).mean()

        stats = {
            "orthogonal_residual_cos": residual_cos.detach(),
            "parallel_norm": parallel_norm.detach(),
            "residual_norm": residual_norm.detach(),
        }

        return pair_loss, stats

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

        user_cf_all, item_cf_all = self.graph_encoder.get_all_embeddings()

        pred = self.predict_from_cf(
            user_id=user_id,
            item_id=item_id,
            user_cf_all=user_cf_all,
            item_cf_all=item_cf_all,
        )

        rating_loss = self.loss_fn(pred, rating)

        user_review_all, item_review_all = self.get_review_graph_embeddings()

        pair_align_loss, pair_stats = self.pair_orthogonal_alignment_loss(
            user_id=user_id,
            item_id=item_id,
            user_cf_all=user_cf_all,
            item_cf_all=item_cf_all,
            user_review_all=user_review_all,
            item_review_all=item_review_all,
        )

        total_loss = rating_loss + self.lambda_pair_align * pair_align_loss

        if return_dict:
            return {
                "loss": total_loss,
                "rating_loss": rating_loss.detach(),
                "pair_align_loss": pair_align_loss.detach(),
                "orthogonal_residual_cos": pair_stats["orthogonal_residual_cos"],
                "parallel_norm": pair_stats["parallel_norm"],
                "residual_norm": pair_stats["residual_norm"],
            }

        return total_loss


