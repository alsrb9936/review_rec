import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.base_model import BaseModel


class RatingGraphEncoder(nn.Module):
    """LightGCN-style encoder over a normalized user-item adjacency matrix."""

    def __init__(self, num_users: int, num_items: int, norm_adj: torch.Tensor, d_id: int, num_layers: int):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.d_id = int(d_id)
        self.num_layers = int(num_layers)

        self.user_embedding = nn.Embedding(self.num_users, self.d_id)
        self.item_embedding = nn.Embedding(self.num_items, self.d_id)
        self.register_buffer("norm_adj", norm_adj.coalesce(), persistent=False)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_all_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_outputs = [x]

        for _ in range(self.num_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            layer_outputs.append(x)

        all_emb = torch.stack(layer_outputs, dim=0).mean(dim=0)
        return all_emb[: self.num_users], all_emb[self.num_users :]

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        user_all, item_all = self.get_all_embeddings()
        return user_all[user_ids], item_all[item_ids]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        *,
        norm_output: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        layers = [
            nn.Linear(int(input_dim), int(hidden_dim)),
            act,
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(output_dim)),
        ]
        if norm_output:
            layers.append(nn.LayerNorm(int(output_dim)))
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReviewEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(output_dim)),
            nn.LayerNorm(int(output_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.net.apply(MLP._init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MyModelV2(BaseModel):
    """
    Clean review-CF hybrid recommender.

    Main scoring path:
      1. CF graph encoder produces user_cf and item_cf.
      2. Review encoders produce user_text and item_text.
      3. Text is projected into CF-conditioned subspaces: user_shared, item_shared.
      4. Pair representations are built from both sides:
           cf_pair     = f([user_cf, item_cf, user_cf * item_cf, |user_cf - item_cf|])
           review_pair = g([user_shared, item_shared, user_shared * item_shared, |user_shared - item_shared|])
      5. Prediction uses cf_score + review_score + bias.

    Contrastive learning:
      - Main: review_pair <-> cf_pair.
      - Optional auxiliary: user_shared <-> user_cf_proj and item_shared <-> item_cf_proj.
    """

    def __init__(self, cfg: DictConfig, norm_adj: torch.Tensor):
        super().__init__(cfg)

        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)

        self.d_id = int(cfg.model.d_id)
        self.d_text = int(cfg.model.d_text)
        self.d_model = int(cfg.model.d_model)
        self.d_pair = int(cfg.model.get("d_pair", self.d_text))
        self.dropout = float(cfg.model.dropout)
        self.num_layers = int(cfg.model.num_layers)
        self.input_dim = int(cfg.data.plm_embedding_size)

        self.eps = float(cfg.model.get("eps", 1e-8))
        self.tau = float(cfg.model.get("contrast_tau", 0.2))
        self.subspace_rank = int(cfg.model.get("subspace_rank", 4))
        self.lambda_pair_align = float(cfg.model.get("lambda_pair_align", 0.0))
        self.lambda_side_align = float(cfg.model.get("lambda_side_align", 0.0))

        if not (0 < self.subspace_rank <= self.d_text):
            raise ValueError(f"subspace_rank must be in [1, d_text], got {self.subspace_rank}")

        self.graph_encoder = RatingGraphEncoder(
            self.num_users,
            self.num_items,
            norm_adj,
            d_id=self.d_id,
            num_layers=self.num_layers,
        )

        self.user_review_encoder = ReviewEncoder(self.input_dim, self.d_text, self.dropout)
        self.item_review_encoder = ReviewEncoder(self.input_dim, self.d_text, self.dropout)

        self.user_basis_layer = nn.Linear(self.d_id, self.d_text * self.subspace_rank)
        self.item_basis_layer = nn.Linear(self.d_id, self.d_text * self.subspace_rank)

        self.cf_pair_encoder = MLP(
            input_dim=4 * self.d_id,
            hidden_dim=self.d_model,
            output_dim=self.d_pair,
            dropout=self.dropout,
            norm_output=True,
        )
        self.review_pair_encoder = MLP(
            input_dim=4 * self.d_text,
            hidden_dim=self.d_model,
            output_dim=self.d_pair,
            dropout=self.dropout,
            norm_output=True,
        )

        self.cf_score_head = nn.Linear(self.d_pair, 1)
        self.review_score_head = nn.Linear(self.d_pair, 1)

        self.user_side_projector = nn.Linear(self.d_id, self.d_text)
        self.item_side_projector = nn.Linear(self.d_id, self.d_text)

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.loss_fn = nn.MSELoss()
        self.apply(self._init_remaining_weights)

    def _init_remaining_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Graph embeddings are initialized inside RatingGraphEncoder.
            if module in {self.user_bias, self.item_bias}:
                nn.init.zeros_(module.weight)

    @staticmethod
    def _pair_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.cat([left, right, left * right, torch.abs(left - right)], dim=-1)

    def _make_basis(self, cf_repr: torch.Tensor, basis_layer: nn.Linear) -> torch.Tensor:
        batch_size = cf_repr.size(0)
        raw_basis = basis_layer(cf_repr).view(batch_size, self.d_text, self.subspace_rank)
        q_basis, _ = torch.linalg.qr(raw_basis, mode="reduced")
        return q_basis

    def _decompose_text(
        self,
        text_repr: torch.Tensor,
        cf_repr: torch.Tensor,
        basis_layer: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_basis = self._make_basis(cf_repr, basis_layer)
        coeff = torch.bmm(q_basis.transpose(1, 2), text_repr.unsqueeze(-1))
        shared = torch.bmm(q_basis, coeff).squeeze(-1)
        residual = text_repr - shared
        return shared, residual

    def _same_id_false_negative_mask(self, ids: torch.Tensor) -> torch.Tensor:
        same_id = ids[:, None].eq(ids[None, :])
        eye = torch.eye(ids.size(0), dtype=torch.bool, device=ids.device)
        return same_id & ~eye

    def _same_pair_false_negative_mask(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        same_user = user_ids[:, None].eq(user_ids[None, :])
        same_item = item_ids[:, None].eq(item_ids[None, :])
        same_pair = same_user & same_item
        eye = torch.eye(user_ids.size(0), dtype=torch.bool, device=user_ids.device)
        return same_pair & ~eye

    def _contrastive_loss(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        false_negative_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if left.size(0) <= 1:
            return left.new_tensor(0.0)

        left = F.normalize(left, dim=-1, eps=self.eps)
        right = F.normalize(right, dim=-1, eps=self.eps)

        logits = left @ right.T
        logits = logits / max(self.tau, self.eps)

        if false_negative_mask is not None:
            logits = logits.masked_fill(false_negative_mask, float("-inf"))

        labels = torch.arange(logits.size(0), device=logits.device)
        loss_left = F.cross_entropy(logits, labels)
        loss_right = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_left + loss_right)

    def encode(self, user_id: torch.Tensor, item_id: torch.Tensor, user_review: torch.Tensor, item_review: torch.Tensor) -> Dict[str, torch.Tensor]:
        user_cf, item_cf = self.graph_encoder(user_id, item_id)

        user_text = self.user_review_encoder(user_review)
        item_text = self.item_review_encoder(item_review)

        user_shared, user_residual = self._decompose_text(user_text, user_cf, self.user_basis_layer)
        item_shared, item_residual = self._decompose_text(item_text, item_cf, self.item_basis_layer)

        cf_pair = self.cf_pair_encoder(self._pair_features(user_cf, item_cf))
        review_pair = self.review_pair_encoder(self._pair_features(user_shared, item_shared))

        cf_score = self.cf_score_head(cf_pair).squeeze(-1)
        review_score = self.review_score_head(review_pair).squeeze(-1)
        bias = (
            self.user_bias(user_id).squeeze(-1)
            + self.item_bias(item_id).squeeze(-1)
            + self.global_bias
        )

        return {
            "user_cf": user_cf,
            "item_cf": item_cf,
            "user_text": user_text,
            "item_text": item_text,
            "user_shared": user_shared,
            "item_shared": item_shared,
            "user_residual": user_residual,
            "item_residual": item_residual,
            "cf_pair": cf_pair,
            "review_pair": review_pair,
            "cf_score": cf_score,
            "review_score": review_score,
            "bias": bias,
            "rating_pred": cf_score + review_score + bias,
        }

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_review: torch.Tensor,
        item_review: torch.Tensor,
        return_dict: bool = False,
    ):
        outputs = self.encode(user_id, item_id, user_review, item_review)
        return outputs if return_dict else outputs["rating_pred"]

    def calculate_loss(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_review: torch.Tensor,
        item_review: torch.Tensor,
        rating: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.forward(user_id, item_id, user_review, item_review, return_dict=True)

        rating_loss = self.loss_fn(outputs["rating_pred"], rating)
        total_loss = rating_loss

        if self.lambda_pair_align > 0.0:
            pair_mask = self._same_pair_false_negative_mask(user_id, item_id)
            pair_align_loss = self._contrastive_loss(outputs["review_pair"], outputs["cf_pair"], pair_mask)
            total_loss = total_loss + self.lambda_pair_align * pair_align_loss

        if self.lambda_side_align > 0.0:
            user_cf_text = self.user_side_projector(outputs["user_cf"])
            item_cf_text = self.item_side_projector(outputs["item_cf"])

            user_mask = self._same_id_false_negative_mask(user_id)
            item_mask = self._same_id_false_negative_mask(item_id)

            side_align_loss = (
                self._contrastive_loss(outputs["user_shared"], user_cf_text, user_mask)
                + self._contrastive_loss(outputs["item_shared"], item_cf_text, item_mask)
            )
            total_loss = total_loss + self.lambda_side_align * side_align_loss

        return total_loss