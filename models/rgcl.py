# models/rgcl.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class ContrastLoss(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.w)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x, y):
        """
        Positive:
            x_i ↔ y_i

        Negative:
            x_i ↔ shuffled(y)_i
        """
        pos_score = (x @ self.w * y).sum(dim=-1)

        perm = torch.randperm(y.size(0), device=y.device)
        y_neg = y[perm]
        neg_score = (x @ self.w * y_neg).sum(dim=-1)

        logits = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(pos_score),
                torch.zeros_like(neg_score),
            ],
            dim=0,
        )

        return self.loss_fn(logits, labels)


class RGCLGraphEncoder(nn.Module):
    def __init__(self, num_users, num_items, review_dim, hidden_dim, dropout):
        super().__init__()

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.review_dim = int(review_dim)
        self.hidden_dim = int(hidden_dim)

        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.hidden_dim)

        self.prob_score = nn.Linear(self.review_dim, 1, bias=False)
        self.review_score = nn.Linear(self.review_dim, 1, bias=False)
        self.review_w = nn.Linear(self.review_dim, self.hidden_dim, bias=False)

        self.dropout = nn.Dropout(float(dropout))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.prob_score.weight)
        nn.init.xavier_uniform_(self.review_score.weight)
        nn.init.xavier_uniform_(self.review_w.weight)

    def forward(self, graph):
        device = self.user_embedding.weight.device

        user_out = torch.zeros(
            self.num_users,
            self.hidden_dim,
            device=device,
        )
        item_out = torch.zeros(
            self.num_items,
            self.hidden_dim,
            device=device,
        )

        user_deg = torch.zeros(self.num_users, 1, device=device)
        item_deg = torch.zeros(self.num_items, 1, device=device)

        for rating in graph["rating_values"]:
            users, items = graph["edge_index_by_rating"][rating]
            review_feat = graph["edge_review_by_rating"][rating]

            users = users.to(device)
            items = items.to(device)
            review_feat = review_feat.to(device)

            user_base = self.user_embedding(users)
            item_base = self.item_embedding(items)

            pa = torch.sigmoid(self.prob_score(review_feat))
            rf = self.review_w(review_feat) * torch.sigmoid(
                self.review_score(review_feat)
            )

            # user -> item message
            msg_item = self.dropout(user_base * pa + rf)
            item_out.index_add_(0, items, msg_item)
            item_deg.index_add_(0, items, torch.ones_like(pa))

            # item -> user message
            msg_user = self.dropout(item_base * pa + rf)
            user_out.index_add_(0, users, msg_user)
            user_deg.index_add_(0, users, torch.ones_like(pa))

        user_out = user_out / user_deg.clamp_min(1.0)
        item_out = item_out / item_deg.clamp_min(1.0)

        return user_out, item_out
    
class RGCL(BaseModel):
    def __init__(self, cfg, rgcl_graph):
        super().__init__(cfg)

        self.cfg = cfg
        self.rgcl_graph = rgcl_graph

        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)
        self.review_dim = int(cfg.data.plm_embedding_size)
        self.hidden_dim = int(cfg.model.hidden_dim)
        self.dropout = float(cfg.model.dropout)

        self.lambda_ed = float(cfg.model.lambda_ed)
        self.lambda_nd = float(cfg.model.lambda_nd)

        self.encoder = RGCLGraphEncoder(
            num_users=self.num_users,
            num_items=self.num_items,
            review_dim=self.review_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        self.pair_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.rating_predictor = nn.Linear(self.hidden_dim, 1)

        self.review_proj = nn.Linear(self.review_dim, self.hidden_dim)

        self.edge_contrast = ContrastLoss(self.hidden_dim)
        self.node_contrast = ContrastLoss(self.hidden_dim)

        self.rating_loss_fn = nn.MSELoss()

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_graph_device(self, device):
        for key in ["users", "items", "ratings", "review_feat"]:
            self.rgcl_graph[key] = self.rgcl_graph[key].to(device)

        for rating in self.rgcl_graph["rating_values"]:
            users, items = self.rgcl_graph["edge_index_by_rating"][rating]
            self.rgcl_graph["edge_index_by_rating"][rating] = (
                users.to(device),
                items.to(device),
            )
            self.rgcl_graph["edge_review_by_rating"][rating] = (
                self.rgcl_graph["edge_review_by_rating"][rating].to(device)
            )

    def encode(self):
        return self.encoder(self.rgcl_graph)

    def make_pair_repr(self, user_emb, item_emb, user_id, item_id):
        u = user_emb[user_id]
        i = item_emb[item_id]

        pair_feat = torch.cat(
            [
                u,
                i,
                u * i,
                torch.abs(u - i),
            ],
            dim=-1,
        )

        return self.pair_proj(pair_feat)

    def decode(self, user_emb, item_emb, user_id, item_id):
        h_ui = self.make_pair_repr(user_emb, item_emb, user_id, item_id)

        pred = self.rating_predictor(h_ui).squeeze(-1)
        pred = (
            pred
            + self.user_bias(user_id).squeeze(-1)
            + self.item_bias(item_id).squeeze(-1)
            + self.global_bias
        )

        return pred, h_ui

    def forward(self, user_id, item_id):
        user_emb, item_emb = self.encode()
        pred, _ = self.decode(user_emb, item_emb, user_id, item_id)
        return pred

    def calculate_loss(self, user_id, item_id, rating, review_feat):
        """
        Train only.

        Uses:
          - rating loss
          - ED loss: edge representation ↔ train review feature
          - ND loss: view1 node embedding ↔ view2 node embedding
        """
        user_emb1, item_emb1 = self.encode()
        user_emb2, item_emb2 = self.encode()

        pred1, edge_h1 = self.decode(user_emb1, item_emb1, user_id, item_id)
        pred2, edge_h2 = self.decode(user_emb2, item_emb2, user_id, item_id)

        rating_loss = (
            self.rating_loss_fn(pred1, rating)
            + self.rating_loss_fn(pred2, rating)
        ) / 2.0

        review_h = self.review_proj(review_feat)

        ed_loss = (
            self.edge_contrast(edge_h1, review_h)
            + self.edge_contrast(edge_h2, review_h)
        ) / 2.0

        nd_user_loss = self.node_contrast(user_emb1, user_emb2)
        nd_item_loss = self.node_contrast(item_emb1, item_emb2)
        nd_loss = (nd_user_loss + nd_item_loss) / 2.0

        total_loss = (
            rating_loss
            + self.lambda_ed * ed_loss
            + self.lambda_nd * nd_loss
        )

        return {
            "loss": total_loss,
            "rating_loss": rating_loss.detach(),
            "ed_loss": ed_loss.detach(),
            "nd_loss": nd_loss.detach(),
        }