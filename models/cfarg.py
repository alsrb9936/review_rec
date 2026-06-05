import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def _infer_review_dim(cfg: DictConfig) -> int:
    data_type = str(cfg.data.get("type", "bert"))
    if data_type.lower() in {"none", "null", ""}:
        data_type = "bert"
    path = os.path.join(str(cfg.data.root), str(cfg.data.dataset), data_type, "user_review_emb.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing user_review_emb.npy for CFARG: {path}")
    return int(np.load(path, mmap_mode="r").shape[1])


def _cfg_bool(node, key: str, default: bool) -> bool:
    value = node.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class CFARG(nn.Module):
    """CF-Aligned Review Gating and its ablation variants.

    Variants:
        cf_only: CF embedding dot-product only.
        review_only: projected review embedding dot-product only.
        fusion: CF + projected review without sample-wise gate.
        gated: CF + gated projected review residual.

    The gated variant is intentionally conservative: review_scale can be fixed,
    gate usage can be regularized, and the MLP gate can be multiplied by an
    explicit cosine-based CF-review alignment factor.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)
        self.embedding_dim = int(cfg.model.embedding_dim)
        self.review_dim = _infer_review_dim(cfg)
        self.dropout = float(cfg.model.get("dropout", 0.0))
        self.variant = str(cfg.model.get("variant", "gated"))
        if self.variant not in {"cf_only", "review_only", "fusion", "gated"}:
            raise ValueError(f"Unsupported CFARG variant: {self.variant}")

        self.gate_bias_init = float(cfg.model.get("gate_bias_init", -2.0))
        self.gate_reg_weight = float(cfg.model.get("gate_reg_weight", 0.0))
        self.review_scale_trainable = _cfg_bool(cfg.model, "review_scale_trainable", False)
        self.detach_cf_for_gate = _cfg_bool(cfg.model, "detach_cf_for_gate", True)
        self.use_align_gate = _cfg_bool(cfg.model, "use_align_gate", True)
        self.align_temperature = float(cfg.model.get("align_temperature", 5.0))
        self.align_threshold = float(cfg.model.get("align_threshold", 0.0))

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.user_review_proj = nn.Linear(self.review_dim, self.embedding_dim)
        self.item_review_proj = nn.Linear(self.review_dim, self.embedding_dim)
        self.user_review_norm = nn.LayerNorm(self.embedding_dim)
        self.item_review_norm = nn.LayerNorm(self.embedding_dim)
        self.user_cf_norm = nn.LayerNorm(self.embedding_dim)
        self.item_cf_norm = nn.LayerNorm(self.embedding_dim)

        review_scale_init = float(cfg.model.get("review_scale_init", 0.1))
        if self.review_scale_trainable:
            self.review_scale = nn.Parameter(torch.tensor(review_scale_init, dtype=torch.float32))
        else:
            self.register_buffer("review_scale", torch.tensor(review_scale_init, dtype=torch.float32))

        gate_hidden_dim = int(cfg.model.get("gate_hidden_dim", self.embedding_dim))
        gate_input_dim = self.embedding_dim * 3 + 1
        self.user_gate = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.item_gate = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(gate_hidden_dim, 1),
        )

        self.loss_fn = nn.MSELoss()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.global_bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.user_gate[-1].bias, self.gate_bias_init)
        nn.init.constant_(self.item_gate[-1].bias, self.gate_bias_init)

    @staticmethod
    def _cosine(cf_emb: torch.Tensor, review_proj: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(cf_emb, review_proj, dim=-1, eps=1e-8).unsqueeze(-1)

    def _make_gate(self, cf_emb: torch.Tensor, review_proj: torch.Tensor, gate_mlp: nn.Module):
        cos = self._cosine(cf_emb, review_proj)
        gate_input = torch.cat([cf_emb, review_proj, cf_emb * review_proj, cos], dim=-1)
        mlp_gate = torch.sigmoid(gate_mlp(gate_input))
        if not self.use_align_gate:
            return mlp_gate
        align_gate = torch.sigmoid(self.align_temperature * (cos - self.align_threshold))
        return mlp_gate * align_gate

    def _cf_gate_view(self, cf_emb: torch.Tensor, norm_layer: nn.Module) -> torch.Tensor:
        cf_for_gate = cf_emb.detach() if self.detach_cf_for_gate else cf_emb
        return norm_layer(cf_for_gate)

    def forward(self, user_id, item_id, user_review=None, item_review=None, return_dict=False):
        user_id = user_id.view(-1).long()
        item_id = item_id.view(-1).long()
        user_cf = self.user_embedding(user_id)
        item_cf = self.item_embedding(item_id)
        batch_size = user_id.numel()

        zero_col = user_cf.new_zeros((batch_size, 1))
        zero_vec = user_cf.new_zeros((batch_size, self.embedding_dim))
        user_proj = zero_vec
        item_proj = zero_vec
        user_injection = zero_vec
        item_injection = zero_vec
        user_cos = zero_col
        item_cos = zero_col

        if self.variant == "cf_only":
            user_z = user_cf
            item_z = item_cf
            user_gate = zero_col
            item_gate = zero_col
        else:
            if user_review is None or item_review is None:
                raise ValueError(f"CFARG variant '{self.variant}' requires user/item review embeddings.")
            user_proj = self.user_review_norm(self.user_review_proj(user_review.float()))
            item_proj = self.item_review_norm(self.item_review_proj(item_review.float()))

            user_cf_gate = self._cf_gate_view(user_cf, self.user_cf_norm)
            item_cf_gate = self._cf_gate_view(item_cf, self.item_cf_norm)
            user_cos = self._cosine(user_cf_gate, user_proj)
            item_cos = self._cosine(item_cf_gate, item_proj)

            if self.variant == "review_only":
                user_gate = torch.ones((batch_size, 1), device=user_id.device, dtype=user_cf.dtype)
                item_gate = torch.ones((batch_size, 1), device=item_id.device, dtype=item_cf.dtype)
                user_injection = self.review_scale * user_gate * user_proj
                item_injection = self.review_scale * item_gate * item_proj
                user_z = user_injection
                item_z = item_injection
            elif self.variant == "fusion":
                user_gate = torch.ones((batch_size, 1), device=user_id.device, dtype=user_cf.dtype)
                item_gate = torch.ones((batch_size, 1), device=item_id.device, dtype=item_cf.dtype)
                user_injection = self.review_scale * user_gate * user_proj
                item_injection = self.review_scale * item_gate * item_proj
                user_z = user_cf + user_injection
                item_z = item_cf + item_injection
            else:
                user_gate = self._make_gate(user_cf_gate, user_proj, self.user_gate)
                item_gate = self._make_gate(item_cf_gate, item_proj, self.item_gate)
                cf_score = torch.sum(user_cf * item_cf, dim=-1) / math.sqrt(float(self.embedding_dim))

                user_review_score = torch.sum(user_gate * user_proj * item_cf, dim=-1) / math.sqrt(float(self.embedding_dim))
                item_review_score = torch.sum(item_gate * item_proj * user_cf, dim=-1) / math.sqrt(float(self.embedding_dim))

                score = cf_score + self.review_scale * (user_review_score + item_review_score)

        score = torch.sum(user_z * item_z, dim=-1) / math.sqrt(float(self.embedding_dim))
        bias = self.global_bias + self.user_bias(user_id).squeeze(-1) + self.item_bias(item_id).squeeze(-1)
        rating_pred = score + bias

        if return_dict:
            user_cf_norm = user_cf.detach().norm(dim=-1)
            item_cf_norm = item_cf.detach().norm(dim=-1)
            user_review_proj_norm = user_proj.detach().norm(dim=-1)
            item_review_proj_norm = item_proj.detach().norm(dim=-1)
            user_injection_norm = user_injection.detach().norm(dim=-1)
            item_injection_norm = item_injection.detach().norm(dim=-1)
            return {
                "rating_pred": rating_pred,
                "user_gate": user_gate.squeeze(-1),
                "item_gate": item_gate.squeeze(-1),
                "review_scale": self.review_scale.detach(),
                "user_cos": user_cos.squeeze(-1).detach(),
                "item_cos": item_cos.squeeze(-1).detach(),
                "user_cf_norm": user_cf_norm,
                "item_cf_norm": item_cf_norm,
                "user_review_proj_norm": user_review_proj_norm,
                "item_review_proj_norm": item_review_proj_norm,
                "user_injection_norm": user_injection_norm,
                "item_injection_norm": item_injection_norm,
                "user_injection_ratio": user_injection_norm / user_cf_norm.clamp_min(1e-8),
                "item_injection_ratio": item_injection_norm / item_cf_norm.clamp_min(1e-8),
                "effective_user_review_weight": (self.review_scale.detach() * user_gate.detach()).squeeze(-1),
                "effective_item_review_weight": (self.review_scale.detach() * item_gate.detach()).squeeze(-1),
            }
        return rating_pred

    def calculate_loss(self, user_id, item_id, rating, user_review=None, item_review=None, return_dict=False):
        outputs = self.forward(user_id, item_id, user_review, item_review, return_dict=True)
        rating_loss = self.loss_fn(outputs["rating_pred"], rating.view(-1).float())
        gate_reg_loss = outputs["rating_pred"].new_tensor(0.0)
        if self.variant == "gated" and self.gate_reg_weight > 0:
            gate_reg_loss = outputs["user_gate"].mean() + outputs["item_gate"].mean()
        loss = rating_loss + self.gate_reg_weight * gate_reg_loss
        if return_dict:
            outputs["loss"] = loss
            outputs["rating_loss"] = rating_loss.detach()
            outputs["gate_reg_loss"] = gate_reg_loss.detach()
            return outputs
        return loss
