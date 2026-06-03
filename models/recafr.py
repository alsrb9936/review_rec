import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAdjEdgeDrop(nn.Module):
    def forward(self, adj: torch.Tensor, keep_rate: float) -> torch.Tensor:
        if keep_rate >= 1.0:
            return adj
        adj = adj.coalesce()
        values = adj.values()
        indices = adj.indices()
        mask = (torch.rand(values.shape[0], device=values.device) + keep_rate).floor().bool()
        return torch.sparse_coo_tensor(
            indices[:, mask],
            values[mask],
            size=adj.shape,
            device=adj.device,
        ).coalesce()


def info_nce_loss(
    query_emb: torch.Tensor,
    key_emb: torch.Tensor,
    all_key_emb: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    query_emb = F.normalize(query_emb, dim=-1)
    key_emb = F.normalize(key_emb, dim=-1)
    all_key_emb = F.normalize(all_key_emb, dim=-1)

    numerator = torch.sum(query_emb * key_emb, dim=-1) / temperature
    denominator = query_emb @ all_key_emb.t() / temperature
    return (-numerator + torch.logsumexp(denominator, dim=-1)).mean()


def _load_view_stack(cfg, names_key: str, default_names: list[str], expected_rows: int, label: str):
    data_type = str(cfg.data.get("type", "bert"))
    if data_type.lower() in {"none", "null", ""}:
        data_type = "bert"
    data_dir = os.path.join(cfg.data.root, cfg.data.dataset, data_type)
    filenames = cfg.model.get(names_key, default_names)

    views = []
    for filename in filenames:
        path = os.path.join(data_dir, str(filename))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing RecAFR {label} view: {path}")
        view = np.load(path).astype(np.float32)
        if view.shape[0] != expected_rows:
            raise ValueError(
                f"{label} view row count mismatch for {path}: "
                f"view={view.shape[0]}, expected={expected_rows}"
            )
        views.append(view)

    if len(views) < 1:
        raise ValueError(f"At least one {label} view is required.")

    dims = {view.shape[1] for view in views}
    if len(dims) != 1:
        raise ValueError(f"All {label} views must have the same dim, got {sorted(dims)}")

    return torch.from_numpy(np.stack(views, axis=0))


class RecAFR(nn.Module):
    """Rating-prediction RecAFR/LightGCN+ model using two semantic views.

    The BERT files ``user_review_emb_s1.npy``/``user_review_emb_s2.npy`` and
    ``item_review_emb_s1.npy``/``item_review_emb_s2.npy`` are kept as separate
    views. Each view is projected by a shared profile MLP and used as a KD target
    for the graph embedding. The rating objective remains MSE.
    """

    def __init__(self, cfg, norm_adj: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)
        self.embedding_dim = int(cfg.model.embedding_dim)
        self.num_layers = int(cfg.model.num_layers)
        self.keep_rate = float(cfg.model.keep_rate)
        self.reg_weight = float(cfg.model.reg_weight)
        self.kd_weight = float(cfg.model.kd_weight)
        self.kd_temperature = float(cfg.model.kd_temperature)

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.edge_dropper = SparseAdjEdgeDrop()
        self.loss_fn = nn.MSELoss()
        self.register_buffer("norm_adj", norm_adj.coalesce(), persistent=False)

        user_views = _load_view_stack(
            cfg,
            names_key="user_profile_views",
            default_names=["user_review_emb_s1.npy", "user_review_emb_s2.npy"],
            expected_rows=self.num_users,
            label="user profile",
        )
        item_views = _load_view_stack(
            cfg,
            names_key="item_profile_views",
            default_names=["item_review_emb_s1.npy", "item_review_emb_s2.npy"],
            expected_rows=self.num_items,
            label="item profile",
        )
        if user_views.shape[-1] != item_views.shape[-1]:
            raise ValueError(
                f"User/item profile dim mismatch: user={user_views.shape[-1]}, item={item_views.shape[-1]}"
            )
        if user_views.shape[0] != item_views.shape[0]:
            raise ValueError(
                f"User/item view count mismatch: user={user_views.shape[0]}, item={item_views.shape[0]}"
            )

        self.view_num = int(user_views.shape[0])
        self.register_buffer("user_profile_views", user_views, persistent=False)
        self.register_buffer("item_profile_views", item_views, persistent=False)

        profile_dim = int(user_views.shape[-1])
        hidden_dim = int(cfg.model.get("profile_hidden_dim", (profile_dim + self.embedding_dim) // 2))
        self.profile_mlp = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.embedding_dim),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        for module in self.profile_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_all_embeddings(self, keep_rate: Optional[float] = None):
        if keep_rate is None:
            keep_rate = self.keep_rate if self.training else 1.0

        adj = self.edge_dropper(self.norm_adj, keep_rate) if self.training else self.norm_adj
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_outputs = [embeddings]

        for _ in range(self.num_layers):
            embeddings = torch.sparse.mm(adj, embeddings)
            layer_outputs.append(embeddings)

        final_embeddings = torch.stack(layer_outputs, dim=0).sum(dim=0)
        return torch.split(final_embeddings, [self.num_users, self.num_items], dim=0)

    def get_profile_view_embeddings(self):
        user_view_embeds = []
        item_view_embeds = []
        for view_idx in range(self.view_num):
            user_view_embeds.append(self.profile_mlp(self.user_profile_views[view_idx]))
            item_view_embeds.append(self.profile_mlp(self.item_profile_views[view_idx]))
        return user_view_embeds, item_view_embeds

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        user_id = user_id.view(-1).long()
        item_id = item_id.view(-1).long()
        user_all, item_all = self.get_all_embeddings(keep_rate=1.0 if not self.training else self.keep_rate)
        user_emb = user_all[user_id]
        item_emb = item_all[item_id]
        rating_pred = torch.sum(user_emb * item_emb, dim=-1)
        rating_pred = rating_pred + self.user_bias(user_id).squeeze(-1)
        rating_pred = rating_pred + self.item_bias(item_id).squeeze(-1)
        return rating_pred + self.global_bias

    def calculate_loss(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        rating: torch.Tensor,
    ):
        user_id = user_id.view(-1).long()
        item_id = item_id.view(-1).long()
        rating = rating.view(-1).float()

        prediction = self.forward(user_id=user_id, item_id=item_id)
        mse_loss = self.loss_fn(prediction, rating)

        reg_loss = self.reg_weight * (
            self.user_embedding(user_id).pow(2).sum()
            + self.item_embedding(item_id).pow(2).sum()
        ) / max(user_id.numel(), 1)

        user_all, item_all = self.get_all_embeddings(keep_rate=1.0)
        user_emb = user_all[user_id]
        item_emb = item_all[item_id]
        user_view_embeds, item_view_embeds = self.get_profile_view_embeddings()

        kd_terms = []
        for user_profile_all, item_profile_all in zip(user_view_embeds, item_view_embeds):
            kd_terms.append(
                info_nce_loss(
                    user_emb,
                    user_profile_all[user_id],
                    user_profile_all,
                    self.kd_temperature,
                )
            )
            kd_terms.append(
                info_nce_loss(
                    item_emb,
                    item_profile_all[item_id],
                    item_profile_all,
                    self.kd_temperature,
                )
            )
        kd_loss = torch.stack(kd_terms).mean() * self.kd_weight

        total_loss = mse_loss + reg_loss + kd_loss
        return total_loss, {
            "loss": float(total_loss.detach().cpu()),
            "mse_loss": float(mse_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "kd_loss": float(kd_loss.detach().cpu()),
        }
