import torch
import torch.nn as nn

from models.rgcl import RGCL as BaseRGCL


class OriginalScaleContrastLoss(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.w)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, x, y, y_neg=None):
        pos_score = (x @ self.w * y).sum(dim=-1)
        pos_loss = self.loss_fn(pos_score, torch.ones_like(pos_score))

        if y_neg is None:
            perm = torch.randperm(y.size(0), device=y.device)
            y_neg = y[perm]

        neg_score = (x @ self.w * y_neg).sum(dim=-1)
        neg_loss = self.loss_fn(neg_score, torch.zeros_like(neg_score))

        return pos_loss + neg_loss


class RGCL(BaseRGCL):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.edge_contrast = OriginalScaleContrastLoss(self.hidden_dim)
        self.node_contrast = OriginalScaleContrastLoss(self.hidden_dim)

    def calculate_loss(self, user_id, item_id, rating, review_feat):
        user_emb1, item_emb1 = self.encode()
        user_emb2, item_emb2 = self.encode()

        logits1, edge_h1 = self.decode_logits(user_emb1, item_emb1, user_id, item_id)
        logits2, edge_h2 = self.decode_logits(user_emb2, item_emb2, user_id, item_id)

        pred1 = self.logits_to_expected_rating(logits1)
        pred2 = self.logits_to_expected_rating(logits2)

        rating_loss = (
            self.rating_loss_fn(pred1, rating)
            + self.rating_loss_fn(pred2, rating)
        ) / 2.0

        review_h = self.review_proj(review_feat)
        ed_loss = (
            self.edge_contrast(edge_h1, review_h).mean()
            + self.edge_contrast(edge_h2, review_h).mean()
        ) / 2.0

        nd_user_loss = self.node_contrast(user_emb1, user_emb2).mean()
        nd_item_loss = self.node_contrast(item_emb1, item_emb2).mean()
        nd_loss = (nd_user_loss + nd_item_loss) / 2.0

        total_loss = rating_loss + self.lambda_ed * ed_loss + self.lambda_nd * nd_loss

        return {
            "loss": total_loss,
            "rating_loss": rating_loss.detach(),
            "ed_loss": ed_loss.detach(),
            "nd_loss": nd_loss.detach(),
        }
