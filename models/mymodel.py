import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from models.base_model import BaseModel


class RatingGraphEncoder(nn.Module):
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
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        outs = [x]
        for _ in range(self.num_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            outs.append(x)
        all_emb = torch.stack(outs, dim=0).mean(dim=0)
        return all_emb[: self.num_users], all_emb[self.num_users :]

    def forward(self, user_ids, item_ids):
        user_all, item_all = self.get_all_embeddings()
        return user_all[user_ids], item_all[item_ids]


class ReviewProjectionEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(output_dim)),
            nn.LayerNorm(int(output_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class MyModel(BaseModel):
    def __init__(self, cfg: DictConfig, norm_adj):
        super().__init__(cfg)

        self.num_users = cfg.stats.num_users
        self.num_items = cfg.stats.num_items
        self.d_id = int(cfg.model.d_id)
        self.d_model = int(cfg.model.d_model)
        self.d_text = int(cfg.model.d_text)
        self.dropout = float(cfg.model.dropout)
        self.num_layers = int(cfg.model.num_layers)
        self.input_dim = int(cfg.data.plm_embedding_size)

        self.eps = float(cfg.model.get("eps", 1e-8))
        self.subspace_rank = int(cfg.model.get("subspace_rank", 4))
        self.align_tau = float(cfg.model.get("align_tau", 0.1))
        self.review_scale_init = float(cfg.model.get("review_scale_init", 0.1))
        self.min_rating = float(cfg.data.get("min_rating", 1.0))
        self.max_rating = float(cfg.data.get("max_rating", 5.0))

        if self.subspace_rank <= 0 or self.subspace_rank > self.d_text:
            raise ValueError(f"Invalid subspace_rank: {self.subspace_rank}")

        self.graph_encoder = RatingGraphEncoder(
            num_users=self.num_users,
            num_items=self.num_items,
            norm_adj=norm_adj,
            d_id=self.d_id,
            num_layers=self.num_layers,
        )

        self.user_review_encoder = ReviewProjectionEncoder(
            input_dim=self.input_dim,
            output_dim=self.d_text,
            dropout=self.dropout,
        )
        self.item_review_encoder = ReviewProjectionEncoder(
            input_dim=self.input_dim,
            output_dim=self.d_text,
            dropout=self.dropout,
        )

        self.cf_feature_dim = self.d_id * 4

        self.cf_pair_layer = nn.Sequential(
            nn.Linear(self.cf_feature_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_text),
            nn.LayerNorm(self.d_text),
        )

        self.cf_predict_layer = nn.Sequential(
            nn.Linear(self.cf_feature_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

        self.review_pair_layer = nn.Sequential(
            nn.Linear(self.d_text * 4, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_text),
            nn.LayerNorm(self.d_text),
        )

        self.basis_layer = nn.Linear(self.d_text, self.d_text * self.subspace_rank)

        self.shared_correction_layer = nn.Sequential(
            nn.Linear(self.d_text * 4, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

        self.gate_layer = nn.Linear(4, 1)
        self.review_scale = nn.Parameter(torch.tensor(self.review_scale_init))

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.loss_fn = nn.MSELoss()
        self._init_weights()

    def _init_linear_block(self, block):
        for m in block.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_weights(self):
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.global_bias)

        self._init_linear_block(self.cf_pair_layer)
        self._init_linear_block(self.cf_predict_layer)
        self._init_linear_block(self.review_pair_layer)
        self._init_linear_block(self.shared_correction_layer)

        nn.init.xavier_uniform_(self.basis_layer.weight)
        if self.basis_layer.bias is not None:
            nn.init.zeros_(self.basis_layer.bias)

        nn.init.zeros_(self.gate_layer.weight)
        nn.init.zeros_(self.gate_layer.bias)

    def _cf_features(self, user_cf, item_cf):
        return torch.cat(
            [user_cf, item_cf, user_cf * item_cf, torch.abs(user_cf - item_cf)],
            dim=-1,
        )

    def _review_pair(self, user_review, item_review):
        if user_review.dim() != 2 or item_review.dim() != 2:
            raise ValueError("user_review and item_review must have shape [B, D].")

        user_text = self.user_review_encoder(user_review)
        item_text = self.item_review_encoder(item_review)

        review_features = torch.cat(
            [user_text, item_text, user_text * item_text, torch.abs(user_text - item_text)],
            dim=-1,
        )
        return self.review_pair_layer(review_features)

    def _make_basis(self, c_ui):
        bsz = c_ui.size(0)
        raw_basis = self.basis_layer(c_ui).view(bsz, self.d_text, self.subspace_rank)
        q_basis, _ = torch.linalg.qr(raw_basis, mode="reduced")
        return q_basis
    
    
    def _orthogonal_decompose(self, z_review, c_ui):
        q_basis = self._make_basis(c_ui)
        coeff = torch.bmm(q_basis.transpose(1, 2), z_review.unsqueeze(-1))
        z_shared = torch.bmm(q_basis, coeff).squeeze(-1)
        z_residual = z_review - z_shared

        total_energy = z_review.pow(2).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        shared_ratio = z_shared.pow(2).sum(dim=-1, keepdim=True) / total_energy
        residual_ratio = z_residual.pow(2).sum(dim=-1, keepdim=True) / total_energy

        return z_shared, z_residual, shared_ratio, residual_ratio

    def _rating_alignment_weight(self, rating):
        center = (self.min_rating + self.max_rating) / 2.0
        half_range = max((self.max_rating - self.min_rating) / 2.0, self.eps)
        clarity = (rating - center).abs() / half_range
        clarity = clarity.clamp(0.0, 1.0)
        return self.align_tau + (1.0 - self.align_tau) * clarity

    def forward(self, user_id, item_id, user_review, item_review, return_dict=False):
        user_cf, item_cf = self.graph_encoder(user_id, item_id)

        cf_features = self._cf_features(user_cf, item_cf)
        c_ui = self.cf_pair_layer(cf_features)

        cf_pred = self.cf_predict_layer(cf_features).squeeze(-1)
        cf_pred = (
            cf_pred
            + self.user_bias(user_id).squeeze(-1)
            + self.item_bias(item_id).squeeze(-1)
            + self.global_bias
        )

        z_review = self._review_pair(user_review, item_review)
        z_shared, z_residual, shared_ratio, residual_ratio = self._orthogonal_decompose(
            z_review=z_review,
            c_ui=c_ui,
        )

        correction_features = torch.cat(
            [z_shared, c_ui, z_shared * c_ui, torch.abs(z_shared - c_ui)],
            dim=-1,
        )
        review_delta = self.shared_correction_layer(correction_features).squeeze(-1)

        cos_shared = F.cosine_similarity(z_shared, c_ui, dim=-1, eps=self.eps).unsqueeze(-1)
        cos_residual = F.cosine_similarity(z_residual, c_ui, dim=-1, eps=self.eps).unsqueeze(-1)

        gate_features = torch.cat(
            [shared_ratio, residual_ratio, cos_shared, cos_residual],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_layer(gate_features)).squeeze(-1)

        rating_pred = cf_pred + self.review_scale * gate * review_delta

        if return_dict:
            return {
                "rating_pred": rating_pred,
                "cf_pred": cf_pred,
                "review_delta": review_delta,
                "gate": gate,
                "shared_ratio": shared_ratio.squeeze(-1),
                "residual_ratio": residual_ratio.squeeze(-1),
                "cos_shared": cos_shared.squeeze(-1),
                "cos_residual": cos_residual.squeeze(-1),
                "review_scale": self.review_scale.detach().expand_as(gate),
                "c_ui": c_ui,
                "z_review": z_review,
                "z_shared": z_shared,
                "z_residual": z_residual,
            }

        return rating_pred

    def calculate_loss(self, user_id, item_id, user_review, item_review, rating):
        outputs = self.forward(
            user_id=user_id,
            item_id=item_id,
            user_review=user_review,
            item_review=item_review,
            return_dict=True,
        )

        pred = outputs["rating_pred"]
        rating_loss = self.loss_fn(pred, rating)
        loss = rating_loss

        return loss
