import os

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


def build_recafr_norm_adj(cfg) -> torch.Tensor:
    data_type = str(cfg.data.get("type", "bert"))
    if data_type.lower() in {"none", "null", ""}:
        data_type = "bert"
    data_dir = os.path.join(cfg.data.root, cfg.data.dataset, data_type)

    user_path = os.path.join(data_dir, "train_user_id.npy")
    item_path = os.path.join(data_dir, "train_item_id.npy")
    if not os.path.exists(user_path):
        raise FileNotFoundError(f"Missing train user file: {user_path}")
    if not os.path.exists(item_path):
        raise FileNotFoundError(f"Missing train item file: {item_path}")

    user_ids = np.load(user_path).astype(np.int64)
    item_ids = np.load(item_path).astype(np.int64)
    num_users = int(cfg.stats.num_users)
    num_items = int(cfg.stats.num_items)
    num_nodes = num_users + num_items
    item_nodes = item_ids + num_users

    rows = np.concatenate([user_ids, item_nodes])
    cols = np.concatenate([item_nodes, user_ids])
    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes)).coalesce()

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    row, col = adj.indices()
    values = adj.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(adj.indices(), values, size=adj.shape).coalesce()


def _load_profile_views(cfg):
    data_type = str(cfg.data.get("type", "bert"))
    if data_type.lower() in {"none", "null", ""}:
        data_type = "bert"
    data_dir = os.path.join(cfg.data.root, cfg.data.dataset, data_type)

    user_view_names = cfg.model.get(
        "user_profile_views",
        ["user_review_emb_s1.npy", "user_review_emb_s2.npy"],
    )
    item_view_names = cfg.model.get(
        "item_profile_views",
        ["item_review_emb_s1.npy", "item_review_emb_s2.npy"],
    )

    user_views = []
    item_views = []
    for filename in user_view_names:
        path = os.path.join(data_dir, str(filename))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing RecAFR user profile view: {path}")
        user_views.append(np.load(path).astype(np.float32))

    for filename in item_view_names:
        path = os.path.join(data_dir, str(filename))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing RecAFR item profile view: {path}")
        item_views.append(np.load(path).astype(np.float32))

    user_profile = np.mean(np.stack(user_views, axis=0), axis=0)
    item_profile = np.mean(np.stack(item_views, axis=0), axis=0)

    if user_profile.shape[0] != int(cfg.stats.num_users):
        raise ValueError(
            f"User profile row count mismatch: profile={user_profile.shape[0]}, "
            f"stats.num_users={int(cfg.stats.num_users)}"
        )
    if item_profile.shape[0] != int(cfg.stats.num_items):
        raise ValueError(
            f"Item profile row count mismatch: profile={item_profile.shape[0]}, "
            f"stats.num_items={int(cfg.stats.num_items)}"
        )
    if user_profile.shape[1] != item_profile.shape[1]:
        raise ValueError(
            f"User/item profile dim mismatch: user={user_profile.shape[1]}, "
            f"item={item_profile.shape[1]}"
        )

    return torch.from_numpy(user_profile), torch.from_numpy(item_profile)


class RecAFR(nn.Module):
    """Rating-prediction RecAFR/LightGCN+ model.

    It predicts explicit ratings with an MSE objective. The ``*_emb_s1.npy`` and
    ``*_emb_s2.npy`` files are used as two semantic views, averaged, projected by
    an MLP, and distilled into graph embeddings with InfoNCE.
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

        user_profile, item_profile = _load_profile_views(cfg)
        self.register_buffer("user_profile", user_profile, persistent=False)
        self.register_buffer("item_profile", item_profile, persistent=False)

        profile_dim = int(user_profile.shape[1])
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

    def get_all_embeddings(self, keep_rate: float | None = None):
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

    def get_profile_embeddings(self):
        return self.profile_mlp(self.user_profile), self.profile_mlp(self.item_profile)

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

        loss_reg = self.reg_weight * (
            self.user_embedding(user_id).pow(2).sum()
            + self.item_embedding(item_id).pow(2).sum()
        ) / max(user_id.numel(), 1)

        user_all, item_all = self.get_all_embeddings(keep_rate=1.0)
        user_profile_all, item_profile_all = self.get_profile_embeddings()
        user_emb = user_all[user_id]
        item_emb = item_all[item_id]
        user_profile = user_profile_all[user_id]
        item_profile = item_profile_all[item_id]

        kd_loss = (
            info_nce_loss(user_emb, user_profile, user_profile_all, self.kd_temperature)
            + info_nce_loss(item_emb, item_profile, item_profile_all, self.kd_temperature)
        ) / 2.0
        kd_loss = self.kd_weight * kd_loss

        total_loss = mse_loss + loss_reg + kd_loss
        loss_dict = {
            "loss": float(total_loss.detach().cpu()),
            "mse_loss": float(mse_loss.detach().cpu()),
            "reg_loss": float(loss_reg.detach().cpu()),
            "kd_loss": float(kd_loss.detach().cpu()),
        }
        return total_loss, loss_dict
