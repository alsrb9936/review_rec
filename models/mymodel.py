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
    """
    Projects precomputed PLM mean-pooled review profiles into the CF comparison
    space. This is intentionally shallow: the PLM embedding is already computed
    before training, so this layer should align dimensions rather than learn a
    new text encoder.
    """

    def __init__(self, input_dim, d_text, dropout):
        super().__init__()
        self.projection = nn.Linear(int(input_dim), int(d_text))
        self.layer_norm = nn.LayerNorm(int(d_text))
        self.dropout = nn.Dropout(float(dropout))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        x = self.projection(x)
        x = self.layer_norm(x)
        return self.dropout(x)


class MyModel(BaseModel):
    """
    Minimal CF-guided orthogonal review decomposition.

    Current dataset assumption:
        user_review: [B, D_plm]
            Mean-pooled historical review profile of the target user.
        item_review: [B, D_plm]
            Mean-pooled historical review profile of the target item.

    The target interaction review should not be included in either profile.
    See dataset/mymodel_dataset.py and set data.retain_rui=false.

    Design principle:
        - CF graph builds the collaborative latent representation.
        - Review profiles are projected once into the same dimension.
        - Review signal is decomposed with respect to the CF direction.
        - The shared component gives a small residual correction.
        - The gate is deterministic: the shared energy ratio. No trainable gate MLP.
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
        self.lambda_align = float(cfg.model.get("lambda_align", 0.0))
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

        # One shared projector keeps user/item review profiles in a common PLM-derived space.
        self.review_projector = ReviewProjectionEncoder(
            input_dim=self.input_dim,
            d_text=self.d_text,
            dropout=self.dropout,
        )

        # CF pair representation used as the decomposition direction.
        self.cf_pair_proj = nn.Sequential(
            nn.Linear(self.d_id * 3, self.d_text),
            nn.LayerNorm(self.d_text),
        )

        # Lightweight review-pair fusion. No large concat MLP.
        self.review_pair_norm = nn.LayerNorm(self.d_text)
        self.review_pair_dropout = nn.Dropout(self.dropout)

        # Shared component produces only a residual correction to the CF prediction.
        # Zero initialization makes the initial model exactly CF-only plus biases.
        self.review_delta_layer = nn.Linear(self.d_text, 1)

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.loss_fn = nn.MSELoss()
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.global_bias)

        for module in self.cf_pair_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.review_delta_layer.weight)
        if self.review_delta_layer.bias is not None:
            nn.init.zeros_(self.review_delta_layer.bias)

    def _build_cf_representation(self, user_cf, item_cf):
        cf_raw = torch.cat([user_cf, item_cf, user_cf * item_cf], dim=-1)
        c_ui = self.cf_pair_proj(cf_raw)
        return c_ui

    def _build_review_pair_representation(self, user_review, item_review):
        if user_review.dim() != 2 or item_review.dim() != 2:
            raise ValueError(
                "This simplified MyModel expects mean-pooled review profiles: "
                "user_review and item_review should both have shape [B, D_plm]."
            )

        user_text = self.review_projector(user_review)
        item_text = self.review_projector(item_review)

        # Elementwise fusion keeps the structure compact while still making the
        # representation pair-specific.
        z_review_pair = user_text + item_text + user_text * item_text
        z_review_pair = self.review_pair_norm(z_review_pair)
        return self.review_pair_dropout(z_review_pair)

    def _orthogonal_decompose(self, z_review_pair, c_ui):
        """
        Decompose review-derived representation with respect to the collaborative
        direction.

        z_shared is the component explainable by CF.
        z_residual is the orthogonal complement, interpreted as review-specific signal.
        """
        c_unit = F.normalize(c_ui, p=2, dim=-1, eps=self.eps)
        scalar_projection = (z_review_pair * c_unit).sum(dim=-1, keepdim=True)
        z_shared = scalar_projection * c_unit
        z_residual = z_review_pair - z_shared

        total_energy = z_review_pair.pow(2).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        shared_energy_ratio = z_shared.pow(2).sum(dim=-1, keepdim=True) / total_energy
        residual_energy_ratio = z_residual.pow(2).sum(dim=-1, keepdim=True) / total_energy

        return z_shared, z_residual, shared_energy_ratio, residual_energy_ratio

    def _rating_alignment_weight(self, rating):
        center = (self.min_rating + self.max_rating) / 2.0
        half_range = max((self.max_rating - self.min_rating) / 2.0, self.eps)
        clarity = (rating - center).abs() / half_range
        clarity = clarity.clamp(0.0, 1.0)
        return self.align_tau + (1.0 - self.align_tau) * clarity

    def forward(self, user_id, item_id, user_review, item_review, return_dict=False):
        user_cf, item_cf = self.graph_encoder(user_id, item_id)

        # Pure CF prediction. This avoids hiding review information inside the CF score.
        cf_pred = (user_cf * item_cf).sum(dim=-1) / math.sqrt(self.d_id)
        cf_pred = (
            cf_pred
            + self.user_bias(user_id).squeeze(-1)
            + self.item_bias(item_id).squeeze(-1)
            + self.global_bias
        )

        c_ui = self._build_cf_representation(user_cf, item_cf)
        z_review_pair = self._build_review_pair_representation(user_review, item_review)

        z_shared, z_residual, shared_ratio, residual_ratio = self._orthogonal_decompose(
            z_review_pair=z_review_pair,
            c_ui=c_ui,
        )

        review_delta = self.review_delta_layer(z_shared).squeeze(-1)

        # Deterministic reliability gate. If most review energy lies in the CF-aligned
        # component, the correction is used more. If residual dominates, it is reduced.
        gate = shared_ratio.squeeze(-1).clamp(0.0, 1.0)
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

        if self.lambda_align <= 0.0:
            return rating_loss

        # Optional diagnostic alignment regularization. Keep disabled by default.
        shared_ratio = outputs["shared_ratio"]
        align_weight = self._rating_alignment_weight(rating)
        align_loss = (align_weight * (1.0 - shared_ratio).pow(2)).mean()

        return rating_loss + self.lambda_align * align_loss
