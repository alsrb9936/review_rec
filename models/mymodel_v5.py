# models/mymodel.py

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
    """
    NeuMF encoder.

    Returns:
        cf_latent = [MF interaction vector || MLP output vector]
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        mf_embedding_size: int,
        mlp_embedding_size: int,
        mlp_hidden_size: list[int],
        dropout: float,
    ):
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
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout))
            input_dim = hidden_dim

        self.mlp_layers = nn.Sequential(*layers)
        self.output_dim = self.mf_embedding_size + input_dim

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_mf_embedding.weight)
        nn.init.xavier_normal_(self.item_mf_embedding.weight)
        nn.init.xavier_normal_(self.user_mlp_embedding.weight)
        nn.init.xavier_normal_(self.item_mlp_embedding.weight)

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
        mf_output = user_mf * item_mf

        user_mlp = self.embedding_dropout(self.user_mlp_embedding(user_id))
        item_mlp = self.embedding_dropout(self.item_mlp_embedding(item_id))

        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        cf_latent = torch.cat([mf_output, mlp_output], dim=-1)

        return cf_latent


class MyModelV5(nn.Module):
    """
    NeuMF + interaction-review pair orthogonal decomposition.

    Removed:
        LightGCN graph propagation
        user-side review branch
        item-side review branch
        user_align_loss
        item_align_loss

    Used:
        NeuMF CF latent
        one interaction review embedding
        pair-level review decomposition by CF latent
        contrastive alignment between CF latent and review shared component

    Main prediction:
        rating_pred = NeuMF prediction from CF latent

    Auxiliary:
        review_emb -> review_latent
        review_latent = review_shared + review_orthogonal
        review_shared is aligned with CF latent by InfoNCE

    Default:
        orthogonal_residual_weight = 0.0
        so CL target uses only review_shared.
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
        self.lambda_pair_align = float(cfg.model.get("lambda_pair_align", 0.01))

        self.orthogonal_residual_weight = float(
            cfg.model.get("orthogonal_residual_weight", 0.0)
        )
        self.orthogonal_eps = float(cfg.model.get("orthogonal_eps", cfg.model.get("eps", 1e-8)))

        data_dir = _get_bert_data_dir(cfg)

        review_path = os.path.join(data_dir, "review_emb.npy")
        if not os.path.exists(review_path):
            raise FileNotFoundError(f"Missing review embedding file: {review_path}")

        review_emb_np = np.load(review_path).astype(np.float32)
        self.review_dim = int(review_emb_np.shape[1])

        # Optional train pair -> review index lookup.
        # This lets the model still run if the trainer does not pass review_emb.
        train_user_path = os.path.join(data_dir, "train_user_id.npy")
        train_item_path = os.path.join(data_dir, "train_item_id.npy")

        if os.path.exists(train_user_path) and os.path.exists(train_item_path):
            train_user = np.load(train_user_path).astype(np.int64)
            train_item = np.load(train_item_path).astype(np.int64)

            if len(train_user) != len(train_item) or len(train_user) != len(review_emb_np):
                raise ValueError(
                    "train_user_id.npy, train_item_id.npy, and review_emb.npy must align "
                    "when review_emb is used as train interaction review embeddings. "
                    f"train_user={len(train_user)}, train_item={len(train_item)}, "
                    f"review_emb={len(review_emb_np)}"
                )

            pair_keys = train_user.astype(np.int64) * np.int64(self.num_items) + train_item.astype(np.int64)
            max_key = int(pair_keys.max()) if len(pair_keys) > 0 else -1

            lookup_size = max(max_key + 1, self.num_users * self.num_items if self.num_users * self.num_items <= 5_000_000 else 0)

            # Avoid huge dense lookup for large datasets.
            # If too large, lookup falls back to dictionary tensors via searchsorted.
            if lookup_size > 0 and lookup_size <= 5_000_000:
                dense_lookup = np.full(lookup_size, -1, dtype=np.int64)
                dense_lookup[pair_keys] = np.arange(len(pair_keys), dtype=np.int64)

                self.register_buffer(
                    "pair_to_review_dense",
                    torch.from_numpy(dense_lookup).long(),
                    persistent=False,
                )
                self.register_buffer(
                    "sorted_pair_keys",
                    torch.empty(0, dtype=torch.long),
                    persistent=False,
                )
                self.register_buffer(
                    "sorted_review_indices",
                    torch.empty(0, dtype=torch.long),
                    persistent=False,
                )
            else:
                order = np.argsort(pair_keys)
                self.register_buffer(
                    "pair_to_review_dense",
                    torch.empty(0, dtype=torch.long),
                    persistent=False,
                )
                self.register_buffer(
                    "sorted_pair_keys",
                    torch.from_numpy(pair_keys[order]).long(),
                    persistent=False,
                )
                self.register_buffer(
                    "sorted_review_indices",
                    torch.from_numpy(order.astype(np.int64)).long(),
                    persistent=False,
                )
        else:
            self.register_buffer(
                "pair_to_review_dense",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "sorted_pair_keys",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "sorted_review_indices",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )

        self.register_buffer(
            "train_review_emb",
            torch.from_numpy(review_emb_np).float(),
            persistent=False,
        )

        self.cf_encoder = NeuMFLatentEncoder(
            num_users=self.num_users,
            num_items=self.num_items,
            mf_embedding_size=self.mf_embedding_size,
            mlp_embedding_size=self.mlp_embedding_size,
            mlp_hidden_size=self.mlp_hidden_size,
            dropout=self.dropout,
        )

        self.cf_dim = int(self.cf_encoder.output_dim)
        self.subspace_rank = int(cfg.model.get("subspace_rank", 4))

        if self.subspace_rank <= 0 or self.subspace_rank > self.cf_dim:
            raise ValueError(
                f"Invalid subspace_rank={self.subspace_rank}. "
                f"It must be in [1, {self.cf_dim}]."
            )

        # CF latent -> CF-conditioned orthogonal basis
        self.cf_basis_layer = nn.Linear(
            self.cf_dim,
            self.cf_dim * self.subspace_rank,
        )
        self.review_encoder = nn.Sequential(
            nn.LayerNorm(self.review_dim),
            nn.Linear(self.review_dim, self.cf_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.cf_dim, self.cf_dim),
            nn.LayerNorm(self.cf_dim),
        )

        self.cf_align_proj = nn.Linear(self.cf_dim, self.cf_dim, bias=False)
        self.review_align_proj = nn.Linear(self.cf_dim, self.cf_dim, bias=False)

        self.predict_layer = nn.Linear(self.cf_dim, 1)

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.loss_fn = nn.MSELoss()

        self._init_weights()

    def _init_weights(self):
        for module in self.review_encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
       
        nn.init.xavier_normal_(self.cf_align_proj.weight)
        nn.init.xavier_normal_(self.review_align_proj.weight)

        nn.init.xavier_normal_(self.predict_layer.weight)
        if self.predict_layer.bias is not None:
            nn.init.zeros_(self.predict_layer.bias)

        nn.init.xavier_normal_(self.cf_basis_layer.weight)
        if self.cf_basis_layer.bias is not None:
            nn.init.zeros_(self.cf_basis_layer.bias)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.global_bias)

    def _lookup_review_emb(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Lookup train interaction review embedding by (user_id, item_id).

        Returns:
            review_emb: [B, review_dim]
            valid_mask: [B], bool
        """
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
            pos_clamped = pos.clamp_max(self.sorted_pair_keys.numel() - 1)

            found = self.sorted_pair_keys[pos_clamped] == pair_keys
            in_range = pos < self.sorted_pair_keys.numel()
            valid_mask = in_range & found

            review_idx = torch.full_like(pair_keys, fill_value=-1)
            review_idx[valid_mask] = self.sorted_review_indices[pos_clamped[valid_mask]]
        else:
            valid_mask = torch.zeros_like(pair_keys, dtype=torch.bool)
            review_idx = torch.full_like(pair_keys, fill_value=-1)

        review_emb = torch.zeros(
            user_id.size(0),
            self.review_dim,
            device=user_id.device,
            dtype=self.train_review_emb.dtype,
        )

        if valid_mask.any():
            review_emb[valid_mask] = self.train_review_emb[review_idx[valid_mask]]

        return review_emb, valid_mask

    def _get_review_input(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        review_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prefer explicit batch review_emb.
        If absent, fallback to train interaction lookup.
        """
        if review_emb is not None:
            if review_emb.dim() != 2:
                raise ValueError("review_emb must have shape [B, review_dim].")
            if review_emb.size(-1) != self.review_dim:
                raise ValueError(
                    f"review_emb dim mismatch: got {review_emb.size(-1)}, "
                    f"expected {self.review_dim}"
                )
            valid_mask = torch.ones(
                review_emb.size(0),
                device=review_emb.device,
                dtype=torch.bool,
            )
            return review_emb.float(), valid_mask

        return self._lookup_review_emb(user_id=user_id, item_id=item_id)

    def _make_cf_basis(self, cf_latent: torch.Tensor) -> torch.Tensor:
        """
        Make CF-conditioned orthonormal basis.

        cf_latent:
            [B, D]

        returns:
            q_basis:
                [B, D, K]
                K-dimensional orthonormal subspace basis.
        """
        bsz = cf_latent.size(0)

        # detach cf_latent for basis generation.
        # This prevents the basis generator path from dominating CF latent updates.
        raw_basis = self.cf_basis_layer(cf_latent.detach())
        raw_basis = raw_basis.view(bsz, self.cf_dim, self.subspace_rank)

        # QR gives an orthonormal basis.
        q_basis, _ = torch.linalg.qr(raw_basis, mode="reduced")

        return q_basis


    def _decompose_review_by_cf(
        self,
        review_latent: torch.Tensor,
        cf_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project review latent onto a CF-conditioned subspace.

        review_latent:
            [B, D]

        cf_latent:
            [B, D]

        shared:
            Projection of review_latent onto CF-conditioned subspace.

        orthogonal:
            Residual component orthogonal to that subspace.
        """
        q_basis = self._make_cf_basis(cf_latent)

        # coeff: [B, K, 1]
        coeff = torch.bmm(
            q_basis.transpose(1, 2),
            review_latent.unsqueeze(-1),
        )

        # shared: [B, D]
        shared = torch.bmm(q_basis, coeff).squeeze(-1)

        # Since q_basis is orthonormal, this residual is orthogonal to the subspace.
        orthogonal = review_latent - shared

        total_energy = review_latent.pow(2).sum(dim=-1, keepdim=True).clamp_min(
            self.orthogonal_eps
        )
        shared_ratio = shared.pow(2).sum(dim=-1, keepdim=True) / total_energy
        orthogonal_ratio = orthogonal.pow(2).sum(dim=-1, keepdim=True) / total_energy

        return shared, orthogonal, shared_ratio, orthogonal_ratio

    def _info_nce(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Symmetric in-batch InfoNCE.
        """
        if valid_mask is not None:
            valid_mask = valid_mask.view(-1).bool()
            query = query[valid_mask]
            key = key[valid_mask]

        if query.size(0) <= 1:
            return query.new_tensor(0.0)

        query = F.normalize(query, dim=-1, eps=self.orthogonal_eps)
        key = F.normalize(key, dim=-1, eps=self.orthogonal_eps)

        logits = query @ key.t()
        logits = logits / max(self.temperature, self.orthogonal_eps)

        labels = torch.arange(logits.size(0), device=logits.device)

        loss_qk = F.cross_entropy(logits, labels)
        loss_kq = F.cross_entropy(logits.t(), labels)

        return 0.5 * (loss_qk + loss_kq)

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        review_emb: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)

        cf_latent = self.cf_encoder(user_id, item_id)

        rating_pred = self.predict_layer(cf_latent).squeeze(-1)
        rating_pred = rating_pred + self.user_bias(user_id).squeeze(-1)
        rating_pred = rating_pred + self.item_bias(item_id).squeeze(-1)
        rating_pred = rating_pred + self.global_bias

        review_input, review_valid_mask = self._get_review_input(
            user_id=user_id,
            item_id=item_id,
            review_emb=review_emb,
        )

        review_latent = self.review_encoder(review_input)

        review_shared, review_orthogonal,shared_ratio ,orthogonal_ratio = self._decompose_review_by_cf(
            review_latent=review_latent,
            cf_latent=cf_latent,
        )

        review_used = (
            review_shared
            + self.orthogonal_residual_weight * review_orthogonal
        )

        residual_cos = F.cosine_similarity(
            review_orthogonal,
            cf_latent.detach(),
            dim=-1,
            eps=self.orthogonal_eps,
        ).abs()

        if return_dict:
            return {
                "rating_pred": rating_pred,
                "cf_latent": cf_latent,
                "review_input": review_input,
                "review_latent": review_latent,
                "review_shared": review_shared,
                "review_orthogonal": review_orthogonal,
                "review_used": review_used,
                "review_valid_mask": review_valid_mask,
                
                "orthogonal_residual_cos": residual_cos.mean(),
                "shared_norm": review_shared.norm(dim=-1).mean(),
                "orthogonal_norm": review_orthogonal.norm(dim=-1).mean(),
                "valid_review_ratio": review_valid_mask.float().mean(),
            }

        return rating_pred

    def calculate_loss(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        rating: torch.Tensor,
        review_emb: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        outputs = self.forward(
            user_id=user_id,
            item_id=item_id,
            review_emb=review_emb,
            return_dict=True,
        )

        rating = rating.view(-1).float()
        rating_loss = self.loss_fn(outputs["rating_pred"], rating)

        cf_q = self.cf_align_proj(outputs["cf_latent"])
        review_k = self.review_align_proj(outputs["review_used"])

        pair_align_loss = self._info_nce(
            query=cf_q,
            key=review_k,
            valid_mask=outputs["review_valid_mask"],
        )

        total_loss = rating_loss + self.lambda_pair_align * pair_align_loss

        if return_dict:
            return {
                "loss": total_loss,
                "rating_loss": rating_loss.detach(),
                "pair_align_loss": pair_align_loss.detach(),
                "orthogonal_residual_cos": outputs["orthogonal_residual_cos"].detach(),
                "shared_norm": outputs["shared_norm"].detach(),
                "orthogonal_norm": outputs["orthogonal_norm"].detach(),
                "valid_review_ratio": outputs["valid_review_ratio"].detach(),
            }

        return total_loss

