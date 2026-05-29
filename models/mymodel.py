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
        self.num_nodes = self.num_users + self.num_items
        self.d_id = int(d_id)
        self.num_layers = int(num_layers)

        self.user_embedding = nn.Embedding(self.num_users, self.d_id)
        self.item_embedding = nn.Embedding(self.num_items, self.d_id)

        self.register_buffer("norm_adj", norm_adj.coalesce())

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_all_embeddings(self):
        x = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0,
        )

        layer_outputs = [x]

        for _ in range(self.num_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            layer_outputs.append(x)

        all_embeddings = torch.stack(layer_outputs, dim=0).mean(dim=0)

        user_embeddings = all_embeddings[: self.num_users]
        item_embeddings = all_embeddings[self.num_users :]

        return user_embeddings, item_embeddings

    def forward(self, user_ids, item_ids):
        user_all, item_all = self.get_all_embeddings()

        user_cf = user_all[user_ids]
        item_cf = item_all[item_ids]

        return user_cf, item_cf


class ReviewProjectionEncoder(nn.Module):
    def __init__(self, input_dim, d_text, dropout):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_text = int(d_text)
        self.dropout = float(dropout)

        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.d_text),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.layer_norm = nn.LayerNorm(self.d_text)
        self._init_weights()

    def _init_weights(self):
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.layer_norm(self.projection(x))


class MyModel(BaseModel):
    """
    CF-guided Orthogonal Review Decomposition for rating prediction.

    Expected input:
        user_id:     [B]
        item_id:     [B]
        user_review: [B, D_plm] or [B, K_u, D_plm]
                     historical reviews written by the target user,
                     excluding the target interaction review.
        item_review: [B, D_plm] or [B, K_i, D_plm]
                     historical reviews received by the target item,
                     excluding the target interaction review.

    The model keeps collaborative and review signals separated:
        1) RatingGraphEncoder builds the collaborative latent representation.
        2) Review encoders build a pair-specific review-derived representation.
        3) The review-derived representation is orthogonally decomposed with
           respect to the collaborative representation.
        4) The shared component is used as a gated rating correction.
        5) The residual component is used to estimate review-signal reliability.
    """

    def __init__(self, cfg: DictConfig, norm_adj):
        super().__init__(cfg)

        self.num_users = cfg.stats.num_users
        self.num_items = cfg.stats.num_items
        self.d_id = cfg.model.d_id
        self.d_model = cfg.model.d_model
        self.d_text = cfg.model.d_text
        self.dropout = cfg.model.dropout
        self.num_layers = cfg.model.num_layers
        self.input_dim = cfg.data.plm_embedding_size

        self.eps = float(cfg.model.get("eps", 1e-8))
        self.lambda_align = float(cfg.model.get("lambda_align", 0.01))
        self.lambda_gate = float(cfg.model.get("lambda_gate", 0.001))
        self.align_tau = float(cfg.model.get("align_tau", 0.1))
        self.min_rating = float(cfg.data.get("min_rating", 1.0))
        self.max_rating = float(cfg.data.get("max_rating", 5.0))

        self.graph_encoder = RatingGraphEncoder(
            num_users=self.num_users,
            num_items=self.num_items,
            norm_adj=norm_adj,
            d_id=self.d_id,
            num_layers=self.num_layers,
        )

        self.user_review_encoder = ReviewProjectionEncoder(
            input_dim=self.input_dim,
            d_text=self.d_text,
            dropout=self.dropout,
        )

        self.item_review_encoder = ReviewProjectionEncoder(
            input_dim=self.input_dim,
            d_text=self.d_text,
            dropout=self.dropout,
        )

        # CF branch: user/item graph embeddings -> pair-level collaborative vector.
        self.cf_pair_proj = nn.Sequential(
            nn.Linear(self.d_id * 3, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_text),
            nn.LayerNorm(self.d_text),
        )

        self.cf_predict_layer = nn.Sequential(
            nn.Linear(self.d_text, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

        # Query projections are used only when review histories are given as
        # [B, K, D]. If the dataset already provides mean-pooled [B, D]
        # review profiles, these layers are not used.
        self.user_query_proj = nn.Linear(self.d_id, self.d_text)
        self.item_query_proj = nn.Linear(self.d_id, self.d_text)

        # Review branch: user-side and item-side review profiles -> pair-specific
        # review-derived representation.
        self.review_pair_layer = nn.Sequential(
            nn.Linear(self.d_text * 3, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_text),
            nn.LayerNorm(self.d_text),
        )

        # Shared review component produces a correction to the CF prediction.
        self.review_delta_layer = nn.Sequential(
            nn.Linear(self.d_text, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

        # Reliability gate uses collaborative, shared, and residual signals.
        self.gate_layer = nn.Sequential(
            nn.Linear(self.d_text * 3 + 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.loss_fn = nn.MSELoss()
        self._init_weights()

    def _init_linear_block(self, block):
        for module in block.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_weights(self):
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.global_bias)

        self._init_linear_block(self.cf_pair_proj)
        self._init_linear_block(self.cf_predict_layer)
        self._init_linear_block(self.review_pair_layer)
        self._init_linear_block(self.review_delta_layer)
        self._init_linear_block(self.gate_layer)

        nn.init.xavier_uniform_(self.user_query_proj.weight)
        nn.init.zeros_(self.user_query_proj.bias)
        nn.init.xavier_uniform_(self.item_query_proj.weight)
        nn.init.zeros_(self.item_query_proj.bias)

    def _encode_review_profile(self, review, encoder, query=None, mask=None):
        """
        Supports both current dataset format and future history-set format.

        Current format:
            review: [B, D_plm]
            returns encoded review profile [B, d_text]

        Future format:
            review: [B, K, D_plm]
            query:  [B, d_text]
            mask:   [B, K], True/1 for valid review, False/0 for padding
            returns query-aware attentive review profile [B, d_text]
        """
        if review.dim() == 2:
            return encoder(review)

        if review.dim() != 3:
            raise ValueError(
                f"review must have shape [B, D] or [B, K, D], got {tuple(review.shape)}"
            )

        encoded = encoder(review)  # [B, K, d_text]

        if query is None:
            if mask is None:
                return encoded.mean(dim=1)

            mask = mask.bool()
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
            pooled = (encoded * mask.unsqueeze(-1).to(encoded.dtype)).sum(dim=1) / denom
            return pooled

        query = F.normalize(query, dim=-1)
        encoded_norm = F.normalize(encoded, dim=-1)
        scores = (encoded_norm * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.d_text)

        if mask is not None:
            mask = mask.bool()
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=1)

        if mask is not None:
            # Avoid NaNs for samples where every history slot is padding.
            empty = ~mask.any(dim=1)
            if empty.any():
                attn = attn.masked_fill(empty.unsqueeze(1), 0.0)

        return torch.bmm(attn.unsqueeze(1), encoded).squeeze(1)

    def _orthogonal_decompose(self, z_review_pair, c_ui):
        """
        Decompose review-derived representation with respect to collaborative
        representation direction.

        z_shared  : component explainable by collaborative subspace.
        z_residual: orthogonal complement, interpreted as review-specific signal.
        """
        c_unit = F.normalize(c_ui, p=2, dim=-1, eps=self.eps)
        scalar_projection = (z_review_pair * c_unit).sum(dim=-1, keepdim=True)
        z_shared = scalar_projection * c_unit
        z_residual = z_review_pair - z_shared

        review_norm = z_review_pair.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        shared_ratio = z_shared.norm(p=2, dim=-1, keepdim=True) / review_norm
        residual_ratio = z_residual.norm(p=2, dim=-1, keepdim=True) / review_norm

        return z_shared, z_residual, shared_ratio, residual_ratio

    def _rating_alignment_weight(self, rating):
        center = (self.min_rating + self.max_rating) / 2.0
        half_range = max((self.max_rating - self.min_rating) / 2.0, self.eps)
        clarity = (rating - center).abs() / half_range
        clarity = clarity.clamp(0.0, 1.0)
        return self.align_tau + (1.0 - self.align_tau) * clarity

    def forward(
        self,
        user_id,
        item_id,
        user_review,
        item_review,
        user_review_mask=None,
        item_review_mask=None,
        return_dict=False,
    ):
        user_cf, item_cf = self.graph_encoder(user_id, item_id)

        cf_raw = torch.cat(
            [user_cf, item_cf, user_cf * item_cf],
            dim=-1,
        )
        c_ui = self.cf_pair_proj(cf_raw)

        cf_pred = self.cf_predict_layer(c_ui).squeeze(-1)
        cf_pred = (
            cf_pred
            + self.user_bias(user_id).squeeze(-1)
            + self.item_bias(item_id).squeeze(-1)
            + self.global_bias
        )

        item_query = self.item_query_proj(item_cf)
        user_query = self.user_query_proj(user_cf)

        user_text = self._encode_review_profile(
            review=user_review,
            encoder=self.user_review_encoder,
            query=item_query,
            mask=user_review_mask,
        )
        item_text = self._encode_review_profile(
            review=item_review,
            encoder=self.item_review_encoder,
            query=user_query,
            mask=item_review_mask,
        )

        z_review_pair = self.review_pair_layer(
            torch.cat([user_text, item_text, user_text * item_text], dim=-1)
        )

        z_shared, z_residual, shared_ratio, residual_ratio = self._orthogonal_decompose(
            z_review_pair=z_review_pair,
            c_ui=c_ui,
        )

        review_delta = self.review_delta_layer(z_shared).squeeze(-1)

        gate_input = torch.cat(
            [c_ui, z_shared, z_residual, shared_ratio, residual_ratio],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_layer(gate_input)).squeeze(-1)

        rating_pred = cf_pred + gate * review_delta

        if return_dict:
            return {
                "rating_pred": rating_pred,
                "cf_pred": cf_pred,
                "review_delta": review_delta,
                "gate": gate,
                "shared_ratio": shared_ratio.squeeze(-1),
                "residual_ratio": residual_ratio.squeeze(-1),
                "c_ui": c_ui,
                "z_review_pair": z_review_pair,
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

        shared_ratio = outputs["shared_ratio"]
        gate = outputs["gate"]

        align_weight = self._rating_alignment_weight(rating)
        align_loss = (align_weight * (1.0 - shared_ratio).pow(2)).mean()

        # Gate should roughly follow how much review information is explained by
        # the collaborative direction, but this is intentionally weak.
        gate_loss = (gate - shared_ratio.detach()).pow(2).mean()

        loss = (
            rating_loss
            + self.lambda_align * align_loss
            + self.lambda_gate * gate_loss
        )

        return loss
