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
        self.lambda_align = float(cfg.model.get("lambda_align", 0.1))
        self.lambda_orth = float(cfg.model.get("lambda_orth", 0.01))
        self.contrast_tau = float(cfg.model.get("contrast_tau", cfg.model.get("align_tau", 0.2)))
        self.alpha_init = float(cfg.model.get("alpha_init", 0.5))
        if self.subspace_rank <= 0 or self.subspace_rank > self.d_text:
            raise ValueError(f"Invalid subspace_rank: {self.subspace_rank}")

        self.graph_encoder = RatingGraphEncoder(self.num_users, self.num_items, norm_adj, self.d_id, self.num_layers)
        self.user_review_encoder = ReviewProjectionEncoder(self.input_dim, self.d_text, self.dropout)
        self.item_review_encoder = ReviewProjectionEncoder(self.input_dim, self.d_text, self.dropout)
        self.cf_feature_dim = self.d_id * 4
        self.cf_pair_layer = nn.Sequential(nn.Linear(self.cf_feature_dim, self.d_model), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(self.d_model, self.d_text), nn.LayerNorm(self.d_text))
        self.cf_predict_layer = nn.Sequential(nn.Linear(self.d_text, self.d_model), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(self.d_model, 1))
        self.review_pair_layer = nn.Sequential(nn.Linear(self.d_text * 2, self.d_model), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(self.d_model, self.d_text), nn.LayerNorm(self.d_text))
        self.basis_layer = nn.Linear(self.d_text, self.d_text * self.subspace_rank)
        self.review_predict_layer = nn.Sequential(nn.Linear(self.d_text, self.d_model), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(self.d_model, 1))
        init = torch.tensor(self.alpha_init, dtype=torch.float32).clamp(self.eps, 1.0 - self.eps)
        self.alpha_logit = nn.Parameter(torch.logit(init))
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
        self._init_linear_block(self.review_predict_layer)
        nn.init.xavier_uniform_(self.basis_layer.weight)
        if self.basis_layer.bias is not None:
            nn.init.zeros_(self.basis_layer.bias)

    def _cf_features(self, user_cf, item_cf):
        return torch.cat([user_cf, item_cf, user_cf * item_cf, torch.abs(user_cf - item_cf)], dim=-1)

    def _review_pair(self, user_review, item_review):
        if user_review.dim() != 2 or item_review.dim() != 2:
            raise ValueError("user_review and item_review must have shape [B, D].")
        user_text = self.user_review_encoder(user_review)
        item_text = self.item_review_encoder(item_review)
        z_review = self.review_pair_layer(torch.cat([user_text, item_text], dim=-1))
        return user_text, item_text, z_review

    def _make_basis(self, c_ui):
        bsz = c_ui.size(0)
        raw_basis = self.basis_layer(c_ui).view(bsz, self.d_text, self.subspace_rank)
        q_basis, _ = torch.linalg.qr(raw_basis, mode="reduced")
        return q_basis

    def _decompose_review(self, z_review, c_ui):
        q_basis = self._make_basis(c_ui)
        coeff = torch.bmm(q_basis.transpose(1, 2), z_review.unsqueeze(-1))
        z_shared = torch.bmm(q_basis, coeff).squeeze(-1)
        z_residual = z_review - z_shared
        total_energy = z_review.pow(2).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        shared_ratio = z_shared.pow(2).sum(dim=-1, keepdim=True) / total_energy
        residual_ratio = z_residual.pow(2).sum(dim=-1, keepdim=True) / total_energy
        return z_shared, z_residual, shared_ratio, residual_ratio

    def _contrastive_loss(self, review_repr, cf_repr):
        if review_repr.size(0) <= 1:
            return review_repr.new_tensor(0.0)
        review_repr = F.normalize(review_repr, dim=-1, eps=self.eps)
        cf_repr = F.normalize(cf_repr, dim=-1, eps=self.eps)
        logits = torch.matmul(review_repr, cf_repr.transpose(0, 1)) / max(self.contrast_tau, self.eps)
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))

    def _orth_loss(self, z_shared, z_residual):
        cos = F.cosine_similarity(z_shared, z_residual, dim=-1, eps=self.eps)
        return cos.pow(2).mean()

    def forward(self, user_id, item_id, user_review, item_review, return_dict=False):
        user_cf, item_cf = self.graph_encoder(user_id, item_id)
        cf_features = self._cf_features(user_cf, item_cf)
        c_ui = self.cf_pair_layer(cf_features)
        cf_signal = self.cf_predict_layer(c_ui).squeeze(-1)
        user_text, item_text, z_review = self._review_pair(user_review, item_review)
        z_shared, z_residual, shared_ratio, residual_ratio = self._decompose_review(z_review, c_ui)
        review_signal = self.review_predict_layer(z_shared).squeeze(-1)
        bias = self.user_bias(user_id).squeeze(-1) + self.item_bias(item_id).squeeze(-1) + self.global_bias
        alpha = torch.sigmoid(self.alpha_logit)
        rating_pred = alpha * cf_signal + (1.0 - alpha) * review_signal + bias
        if return_dict:
            return {
                "rating_pred": rating_pred,
                "cf_signal": cf_signal,
                "review_signal": review_signal,
                "bias": bias,
                "alpha": alpha.detach().expand_as(rating_pred),
                "c_ui": c_ui,
                "user_text": user_text,
                "item_text": item_text,
                "z_review": z_review,
                "z_shared": z_shared,
                "z_residual": z_residual,
                "shared_ratio": shared_ratio.squeeze(-1),
                "residual_ratio": residual_ratio.squeeze(-1),
            }
        return rating_pred

    def calculate_loss(self, user_id, item_id, user_review, item_review, rating):
        outputs = self.forward(user_id=user_id, item_id=item_id, user_review=user_review, item_review=item_review, return_dict=True)
        rating_loss = self.loss_fn(outputs["rating_pred"], rating)
        align_loss = self._contrastive_loss(outputs["z_shared"], outputs["c_ui"])
        orth_loss = self._orth_loss(outputs["z_shared"], outputs["z_residual"])
        return rating_loss + self.lambda_align * align_loss + self.lambda_orth * orth_loss
