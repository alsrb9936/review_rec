# models/mymodel_v5.py

import os
from typing import Optional

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


class NeuMFLatentEncoder(nn.Module):
    def __init__(self, num_users: int, num_items: int, mf_embedding_size: int, mlp_embedding_size: int, mlp_hidden_size: list[int], dropout: float):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.mf_embedding_size = int(mf_embedding_size)
        self.mlp_embedding_size = int(mlp_embedding_size)
        self.mlp_hidden_size = [int(x) for x in mlp_hidden_size]
        self.dropout = float(dropout)

        self.user_mf_embedding = nn.Embedding(self.num_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.num_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.num_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.num_items, self.mlp_embedding_size)
        self.embedding_dropout = nn.Dropout(p=self.dropout)

        layers = []
        input_dim = self.mlp_embedding_size * 2
        for hidden_dim in self.mlp_hidden_size:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=self.dropout)])
            input_dim = hidden_dim
        self.mlp_layers = nn.Sequential(*layers)
        self.output_dim = self.mf_embedding_size + input_dim
        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_mf_embedding, self.item_mf_embedding, self.user_mlp_embedding, self.item_mlp_embedding]:
            nn.init.xavier_normal_(emb.weight)
        for module in self.mlp_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)
        user_mf = self.embedding_dropout(self.user_mf_embedding(user_id))
        item_mf = self.embedding_dropout(self.item_mf_embedding(item_id))
        user_mlp = self.embedding_dropout(self.user_mlp_embedding(user_id))
        item_mlp = self.embedding_dropout(self.item_mlp_embedding(item_id))
        mf_output = user_mf * item_mf
        mlp_output = self.mlp_layers(torch.cat([user_mlp, item_mlp], dim=-1))
        return torch.cat([mf_output, mlp_output], dim=-1)


class MyModelV5(nn.Module):
    """
    NeuMF + learnable selective review alignment.

    The review branch decomposes review_latent into:
      review_shared   : preference-aligned component for CF alignment
      review_residual : review-specific component, preserved but decorrelated from CF

    Final rating prediction remains CF-only, so target reviews are not required at test time.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)

        self.mf_embedding_size = int(cfg.model.get("mf_embedding_size", cfg.model.get("embedding_dim", 32)))
        self.mlp_embedding_size = int(cfg.model.get("mlp_embedding_size", cfg.model.get("embedding_dim", 32)))
        self.mlp_hidden_size = list(cfg.model.get("mlp_hidden_size", [64, 32, 16, 8]))
        self.dropout = float(cfg.model.get("dropout", 0.1))
        self.temperature = float(cfg.model.get("temperature", cfg.model.get("contrast_tau", 0.2)))
        self.eps = float(cfg.model.get("eps", cfg.model.get("orthogonal_eps", 1e-8)))
        self.orthogonal_eps = self.eps

        self.lambda_pair_align = float(cfg.model.get("lambda_pair_align", 0.01))
        self.lambda_shared_rating = float(cfg.model.get("lambda_shared_rating", 0.01))
        self.lambda_reconstruct = float(cfg.model.get("lambda_reconstruct", 0.05))
        self.lambda_residual_cf_decor = float(cfg.model.get("lambda_residual_cf_decor", 0.01))
        self.lambda_shared_residual_decor = float(cfg.model.get("lambda_shared_residual_decor", 0.005))
        self.lambda_gate_balance = float(cfg.model.get("lambda_gate_balance", 0.001))
        self.lambda_component_var = float(cfg.model.get("lambda_component_var", 0.001))
        self.gate_target = float(cfg.model.get("gate_target", 0.5))
        self.min_component_std = float(cfg.model.get("min_component_std", 0.05))
        self.orthogonal_residual_weight = float(cfg.model.get("orthogonal_residual_weight", 0.0))

        data_dir = _get_bert_data_dir(cfg)
        review_path = os.path.join(data_dir, "review_emb.npy")
        if not os.path.exists(review_path):
            raise FileNotFoundError(f"Missing review embedding file: {review_path}")
        review_emb_np = np.load(review_path).astype(np.float32)
        self.review_dim = int(review_emb_np.shape[1])
        self._build_review_lookup(data_dir, review_emb_np)

        self.cf_encoder = NeuMFLatentEncoder(self.num_users, self.num_items, self.mf_embedding_size, self.mlp_embedding_size, self.mlp_hidden_size, self.dropout)
        self.cf_dim = int(self.cf_encoder.output_dim)

        self.review_encoder = nn.Sequential(nn.LayerNorm(self.review_dim), nn.Linear(self.review_dim, self.cf_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(self.cf_dim, self.cf_dim), nn.LayerNorm(self.cf_dim))
        self.selective_gate = nn.Sequential(nn.LayerNorm(self.cf_dim * 2), nn.Linear(self.cf_dim * 2, self.cf_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(self.cf_dim, self.cf_dim))
        self.shared_refiner = nn.Sequential(nn.LayerNorm(self.cf_dim), nn.Linear(self.cf_dim, self.cf_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(self.cf_dim, self.cf_dim), nn.LayerNorm(self.cf_dim))
        self.residual_refiner = nn.Sequential(nn.LayerNorm(self.cf_dim), nn.Linear(self.cf_dim, self.cf_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(self.cf_dim, self.cf_dim), nn.LayerNorm(self.cf_dim))
        self.review_decoder = nn.Sequential(nn.LayerNorm(self.cf_dim * 2), nn.Linear(self.cf_dim * 2, self.cf_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(self.cf_dim, self.review_dim))

        self.cf_align_proj = nn.Linear(self.cf_dim, self.cf_dim, bias=False)
        self.review_align_proj = nn.Linear(self.cf_dim, self.cf_dim, bias=False)
        self.predict_layer = nn.Linear(self.cf_dim, 1)
        self.shared_rating_head = nn.Linear(self.cf_dim, 1)
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.loss_fn = nn.MSELoss()
        self._init_weights()

    def _build_review_lookup(self, data_dir: str, review_emb_np: np.ndarray):
        self.register_buffer("train_review_emb", torch.from_numpy(review_emb_np).float(), persistent=False)
        train_user_path = os.path.join(data_dir, "train_user_id.npy")
        train_item_path = os.path.join(data_dir, "train_item_id.npy")
        empty = torch.empty(0, dtype=torch.long)
        if not (os.path.exists(train_user_path) and os.path.exists(train_item_path)):
            self.register_buffer("pair_to_review_dense", empty, persistent=False)
            self.register_buffer("sorted_pair_keys", empty, persistent=False)
            self.register_buffer("sorted_review_indices", empty, persistent=False)
            return

        train_user = np.load(train_user_path).astype(np.int64)
        train_item = np.load(train_item_path).astype(np.int64)
        if len(train_user) != len(train_item) or len(train_user) != len(review_emb_np):
            raise ValueError("train_user_id.npy, train_item_id.npy, and review_emb.npy must align.")

        pair_keys = train_user * np.int64(self.num_items) + train_item
        max_key = int(pair_keys.max()) if len(pair_keys) > 0 else -1
        lookup_size = max(max_key + 1, self.num_users * self.num_items if self.num_users * self.num_items <= 5_000_000 else 0)
        if lookup_size > 0 and lookup_size <= 5_000_000:
            dense_lookup = np.full(lookup_size, -1, dtype=np.int64)
            dense_lookup[pair_keys] = np.arange(len(pair_keys), dtype=np.int64)
            self.register_buffer("pair_to_review_dense", torch.from_numpy(dense_lookup).long(), persistent=False)
            self.register_buffer("sorted_pair_keys", empty, persistent=False)
            self.register_buffer("sorted_review_indices", empty, persistent=False)
        else:
            order = np.argsort(pair_keys)
            self.register_buffer("pair_to_review_dense", empty, persistent=False)
            self.register_buffer("sorted_pair_keys", torch.from_numpy(pair_keys[order]).long(), persistent=False)
            self.register_buffer("sorted_review_indices", torch.from_numpy(order.astype(np.int64)).long(), persistent=False)

    def _init_weights(self):
        for block in [self.review_encoder, self.selective_gate, self.shared_refiner, self.residual_refiner, self.review_decoder]:
            for module in block:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        for layer in [self.cf_align_proj, self.review_align_proj, self.predict_layer, self.shared_rating_head]:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.global_bias)

    def _masked_rows(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor]):
        return x if valid_mask is None else x[valid_mask.view(-1).bool()]

    def _masked_mean(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        x = self._masked_rows(x, valid_mask)
        return x.new_tensor(0.0) if x.numel() == 0 else x.mean()

    def _lookup_review_emb(self, user_id: torch.Tensor, item_id: torch.Tensor):
        user_id = user_id.view(-1).long()
        item_id = item_id.view(-1).long()
        pair_keys = user_id * int(self.num_items) + item_id
        if self.pair_to_review_dense.numel() > 0:
            in_range = pair_keys < self.pair_to_review_dense.numel()
            safe_keys = pair_keys.clamp_max(self.pair_to_review_dense.numel() - 1)
            review_idx = self.pair_to_review_dense[safe_keys]
            valid_mask = in_range & (review_idx >= 0)
        elif self.sorted_pair_keys.numel() > 0:
            pos = torch.searchsorted(self.sorted_pair_keys, pair_keys)
            in_range = pos < self.sorted_pair_keys.numel()
            pos_clamped = pos.clamp_max(self.sorted_pair_keys.numel() - 1)
            found = torch.zeros_like(in_range, dtype=torch.bool)
            found[in_range] = self.sorted_pair_keys[pos_clamped[in_range]] == pair_keys[in_range]
            valid_mask = in_range & found
            review_idx = torch.full_like(pair_keys, -1)
            review_idx[valid_mask] = self.sorted_review_indices[pos_clamped[valid_mask]]
        else:
            valid_mask = torch.zeros_like(pair_keys, dtype=torch.bool)
            review_idx = torch.full_like(pair_keys, -1)
        review_emb = torch.zeros(user_id.size(0), self.review_dim, device=user_id.device, dtype=self.train_review_emb.dtype)
        if valid_mask.any():
            review_emb[valid_mask] = self.train_review_emb[review_idx[valid_mask]]
        return review_emb, valid_mask

    def _get_review_input(self, user_id: torch.Tensor, item_id: torch.Tensor, review_emb: Optional[torch.Tensor] = None):
        if review_emb is not None:
            if review_emb.dim() != 2:
                raise ValueError("review_emb must have shape [B, review_dim].")
            if review_emb.size(-1) != self.review_dim:
                raise ValueError(f"review_emb dim mismatch: got {review_emb.size(-1)}, expected {self.review_dim}")
            return review_emb.float(), torch.ones(review_emb.size(0), device=review_emb.device, dtype=torch.bool)
        return self._lookup_review_emb(user_id, item_id)

    def _decompose_review_selectively(self, review_latent: torch.Tensor, cf_latent: torch.Tensor):
        gate = torch.sigmoid(self.selective_gate(torch.cat([review_latent, cf_latent.detach()], dim=-1)))
        review_shared = self.shared_refiner(gate * review_latent)
        review_residual = self.residual_refiner((1.0 - gate) * review_latent)
        review_recon = self.review_decoder(torch.cat([review_shared, review_residual], dim=-1))
        return review_shared, review_residual, gate, review_recon

    def _info_nce(self, query: torch.Tensor, key: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        query = self._masked_rows(query, valid_mask)
        key = self._masked_rows(key, valid_mask)
        if query.size(0) <= 1:
            return query.new_tensor(0.0)
        query = F.normalize(query, dim=-1, eps=self.eps)
        key = F.normalize(key, dim=-1, eps=self.eps)
        logits = query @ key.t() / max(self.temperature, self.eps)
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def _cosine_square_loss(self, x: torch.Tensor, y: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        x = self._masked_rows(x, valid_mask)
        y = self._masked_rows(y, valid_mask)
        if x.size(0) == 0:
            return x.new_tensor(0.0)
        return F.cosine_similarity(x, y, dim=-1, eps=self.eps).pow(2).mean()

    def _component_variance_loss(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        x = self._masked_rows(x, valid_mask)
        if x.size(0) <= 1:
            return x.new_tensor(0.0)
        std = torch.sqrt((x - x.mean(dim=0, keepdim=True)).var(dim=0, unbiased=False) + self.eps)
        return F.relu(self.min_component_std - std).mean()

    def _reconstruction_loss(self, review_recon: torch.Tensor, review_input: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        review_recon = self._masked_rows(review_recon, valid_mask)
        review_input = self._masked_rows(review_input, valid_mask)
        if review_recon.size(0) == 0:
            return review_recon.new_tensor(0.0)
        return F.mse_loss(F.normalize(review_recon, dim=-1, eps=self.eps), F.normalize(review_input, dim=-1, eps=self.eps))

    def _gate_balance_loss(self, gate: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        gate = self._masked_rows(gate, valid_mask)
        if gate.size(0) == 0:
            return gate.new_tensor(0.0)
        return (gate.mean() - self.gate_target).pow(2)

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor, review_emb: Optional[torch.Tensor] = None, return_dict: bool = False):
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)
        cf_latent = self.cf_encoder(user_id, item_id)
        rating_pred = self.predict_layer(cf_latent).squeeze(-1) + self.user_bias(user_id).squeeze(-1) + self.item_bias(item_id).squeeze(-1) + self.global_bias

        review_input, review_valid_mask = self._get_review_input(user_id, item_id, review_emb)
        review_latent = self.review_encoder(review_input)
        review_shared, review_residual, gate, review_recon = self._decompose_review_selectively(review_latent, cf_latent)
        review_used = review_shared + self.orthogonal_residual_weight * review_residual
        shared_rating_pred = self.shared_rating_head(review_shared).squeeze(-1)

        if not return_dict:
            return rating_pred

        residual_cf_cos = F.cosine_similarity(review_residual, cf_latent.detach(), dim=-1, eps=self.eps).abs()
        shared_cf_cos = F.cosine_similarity(review_shared, cf_latent.detach(), dim=-1, eps=self.eps)
        shared_residual_cos = F.cosine_similarity(review_shared, review_residual, dim=-1, eps=self.eps).abs()
        return {
            "rating_pred": rating_pred,
            "shared_rating_pred": shared_rating_pred,
            "cf_latent": cf_latent,
            "review_input": review_input,
            "review_latent": review_latent,
            "review_shared": review_shared,
            "review_residual": review_residual,
            "review_orthogonal": review_residual,
            "review_used": review_used,
            "review_recon": review_recon,
            "review_gate": gate,
            "review_valid_mask": review_valid_mask,
            "residual_cf_cos": self._masked_mean(residual_cf_cos, review_valid_mask),
            "orthogonal_residual_cos": self._masked_mean(residual_cf_cos, review_valid_mask),
            "shared_cf_cos": self._masked_mean(shared_cf_cos, review_valid_mask),
            "shared_residual_cos": self._masked_mean(shared_residual_cos, review_valid_mask),
            "shared_norm": self._masked_mean(review_shared.norm(dim=-1), review_valid_mask),
            "residual_norm": self._masked_mean(review_residual.norm(dim=-1), review_valid_mask),
            "orthogonal_norm": self._masked_mean(review_residual.norm(dim=-1), review_valid_mask),
            "gate_mean": self._masked_mean(gate.mean(dim=-1), review_valid_mask),
            "valid_review_ratio": review_valid_mask.float().mean(),
        }

    def calculate_loss(self, user_id: torch.Tensor, item_id: torch.Tensor, rating: torch.Tensor, review_emb: Optional[torch.Tensor] = None, return_dict: bool = False):
        outputs = self.forward(user_id, item_id, review_emb=review_emb, return_dict=True)
        rating = rating.view(-1).float()
        valid_mask = outputs["review_valid_mask"]

        rating_loss = self.loss_fn(outputs["rating_pred"], rating)
        pair_align_loss = self._info_nce(self.cf_align_proj(outputs["cf_latent"]), self.review_align_proj(outputs["review_used"]), valid_mask)
        shared_rating_loss = F.mse_loss(outputs["shared_rating_pred"][valid_mask], rating[valid_mask]) if valid_mask.any() else rating.new_tensor(0.0)
        reconstruct_loss = self._reconstruction_loss(outputs["review_recon"], outputs["review_input"], valid_mask)
        residual_cf_decor_loss = self._cosine_square_loss(outputs["review_residual"], outputs["cf_latent"].detach(), valid_mask)
        shared_residual_decor_loss = self._cosine_square_loss(outputs["review_shared"], outputs["review_residual"], valid_mask)
        gate_balance_loss = self._gate_balance_loss(outputs["review_gate"], valid_mask)
        component_var_loss = 0.5 * (self._component_variance_loss(outputs["review_shared"], valid_mask) + self._component_variance_loss(outputs["review_residual"], valid_mask))

        total_loss = (
            rating_loss
            + self.lambda_pair_align * pair_align_loss
            + self.lambda_shared_rating * shared_rating_loss
            + self.lambda_reconstruct * reconstruct_loss
            + self.lambda_residual_cf_decor * residual_cf_decor_loss
            + self.lambda_shared_residual_decor * shared_residual_decor_loss
            + self.lambda_gate_balance * gate_balance_loss
            + self.lambda_component_var * component_var_loss
        )
        if not return_dict:
            return total_loss
        return {
            "loss": total_loss,
            "rating_loss": rating_loss.detach(),
            "pair_align_loss": pair_align_loss.detach(),
            "shared_rating_loss": shared_rating_loss.detach(),
            "reconstruct_loss": reconstruct_loss.detach(),
            "residual_cf_decor_loss": residual_cf_decor_loss.detach(),
            "shared_residual_decor_loss": shared_residual_decor_loss.detach(),
            "gate_balance_loss": gate_balance_loss.detach(),
            "component_var_loss": component_var_loss.detach(),
            "residual_cf_cos": outputs["residual_cf_cos"].detach(),
            "orthogonal_residual_cos": outputs["orthogonal_residual_cos"].detach(),
            "shared_cf_cos": outputs["shared_cf_cos"].detach(),
            "shared_residual_cos": outputs["shared_residual_cos"].detach(),
            "shared_norm": outputs["shared_norm"].detach(),
            "residual_norm": outputs["residual_norm"].detach(),
            "orthogonal_norm": outputs["orthogonal_norm"].detach(),
            "gate_mean": outputs["gate_mean"].detach(),
            "valid_review_ratio": outputs["valid_review_ratio"].detach(),
        }


MyModel = MyModelV5
MyModelV4 = MyModelV5
