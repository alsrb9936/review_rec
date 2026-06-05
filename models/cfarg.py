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


class CFARG(nn.Module):
    """CF-Aligned Review Gating and its ablation variants."""

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
        self.review_scale = nn.Parameter(
            torch.tensor(float(cfg.model.get("review_scale_init", 0.1)))
        )

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

    def _make_gate(self, cf_emb: torch.Tensor, review_proj: torch.Tensor, gate_mlp: nn.Module):
        cos = F.cosine_similarity(cf_emb, review_proj, dim=-1, eps=1e-8).unsqueeze(-1)
        gate_input = torch.cat([cf_emb, review_proj, cf_emb * review_proj, cos], dim=-1)
        return torch.sigmoid(gate_mlp(gate_input))

    def forward(self, user_id, item_id, user_review=None, item_review=None, return_dict=False):
        user_id = user_id.view(-1).long()
        item_id = item_id.view(-1).long()
        user_cf = self.user_embedding(user_id)
        item_cf = self.item_embedding(item_id)

        if self.variant == "cf_only":
            user_z = user_cf
            item_z = item_cf
            user_gate = torch.zeros((len(user_id), 1), device=user_id.device)
            item_gate = torch.zeros((len(item_id), 1), device=item_id.device)
        else:
            if user_review is None or item_review is None:
                raise ValueError(f"CFARG variant '{self.variant}' requires user/item review embeddings.")
            user_proj = self.user_review_norm(self.user_review_proj(user_review.float()))
            item_proj = self.item_review_norm(self.item_review_proj(item_review.float()))

            if self.variant == "review_only":
                user_z = self.review_scale * user_proj
                item_z = self.review_scale * item_proj
                user_gate = torch.ones((len(user_id), 1), device=user_id.device)
                item_gate = torch.ones((len(item_id), 1), device=item_id.device)
            elif self.variant == "fusion":
                user_z = user_cf + self.review_scale * user_proj
                item_z = item_cf + self.review_scale * item_proj
                user_gate = torch.ones((len(user_id), 1), device=user_id.device)
                item_gate = torch.ones((len(item_id), 1), device=item_id.device)
            else:
                user_cf_gate = self.user_cf_norm(user_cf)
                item_cf_gate = self.item_cf_norm(item_cf)
                user_gate = self._make_gate(user_cf_gate, user_proj, self.user_gate)
                item_gate = self._make_gate(item_cf_gate, item_proj, self.item_gate)
                user_z = user_cf + self.review_scale * user_gate * user_proj
                item_z = item_cf + self.review_scale * item_gate * item_proj

        score = torch.sum(user_z * item_z, dim=-1) / math.sqrt(float(self.embedding_dim))
        bias = self.global_bias + self.user_bias(user_id).squeeze(-1) + self.item_bias(item_id).squeeze(-1)
        rating_pred = score + bias
        if return_dict:
            return {
                "rating_pred": rating_pred,
                "user_gate": user_gate.squeeze(-1),
                "item_gate": item_gate.squeeze(-1),
                "review_scale": self.review_scale.detach(),
            }
        return rating_pred

    def calculate_loss(self, user_id, item_id, rating, user_review=None, item_review=None, return_dict=False):
        outputs = self.forward(user_id, item_id, user_review, item_review, return_dict=True)
        loss = self.loss_fn(outputs["rating_pred"], rating.view(-1).float())
        if return_dict:
            outputs["loss"] = loss
            return outputs
        return loss
