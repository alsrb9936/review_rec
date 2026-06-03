# models/rgcl.py
import torch
import torch.nn as nn
import dgl.function as fn
import dgl.nn.pytorch as dglnn

from utils.rgcl_graph import build_rgcl_graph_from_train, infer_review_dim, rating_to_etype_name


class ContrastLoss(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.w)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x, y):
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


class ReviewAwareGraphConv(nn.Module):
    """Rating-specific DGL relation module used inside HeteroGraphConv."""

    def __init__(self, num_src_nodes: int, review_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.num_src_nodes = int(num_src_nodes)
        self.review_dim = int(review_dim)
        self.hidden_dim = int(hidden_dim)

        # Same structure as ReviewGraph: each rating relation owns its own
        # source-node free embedding matrix.
        self.weight = nn.Parameter(torch.empty(self.num_src_nodes, self.hidden_dim))
        self.prob_score = nn.Linear(self.review_dim, 1, bias=False)
        self.review_score = nn.Linear(self.review_dim, 1, bias=False)
        self.review_w = nn.Linear(self.review_dim, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.prob_score.weight)
        nn.init.xavier_uniform_(self.review_score.weight)
        nn.init.xavier_uniform_(self.review_w.weight)

    def forward(self, graph, feat=None):
        with graph.local_scope():
            graph.srcdata["h"] = self.weight
            review_feat = graph.edata["review_feat"]

            graph.edata["pa"] = torch.sigmoid(self.prob_score(review_feat))
            graph.edata["rf"] = self.review_w(review_feat) * torch.sigmoid(
                self.review_score(review_feat)
            )

            def message_func(edges):
                message = edges.src["h"] * edges.data["pa"] + edges.data["rf"]
                message = self.dropout(message) * edges.src["cj"]
                return {"m": message}

            graph.update_all(message_func, fn.sum(msg="m", out="h"))
            return graph.dstdata["h"] * graph.dstdata["ci"]


class RGCLGraphEncoder(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        review_dim,
        hidden_dim,
        dropout,
        rating_values,
    ):
        super().__init__()

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.review_dim = int(review_dim)
        self.hidden_dim = int(hidden_dim)
        self.rating_values = [float(r) for r in rating_values]
        self.etype_names = [rating_to_etype_name(r) for r in self.rating_values]

        sub_conv = {}
        for etype in self.etype_names:
            sub_conv[etype] = ReviewAwareGraphConv(
                num_src_nodes=self.num_users,
                review_dim=self.review_dim,
                hidden_dim=self.hidden_dim,
                dropout=dropout,
            )
            sub_conv[f"rev-{etype}"] = ReviewAwareGraphConv(
                num_src_nodes=self.num_items,
                review_dim=self.review_dim,
                hidden_dim=self.hidden_dim,
                dropout=dropout,
            )

        self.conv = dglnn.HeteroGraphConv(sub_conv, aggregate="sum")
        self.user_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.item_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_fc.weight)
        nn.init.xavier_uniform_(self.item_fc.weight)
        nn.init.zeros_(self.user_fc.bias)
        nn.init.zeros_(self.item_fc.bias)

    def forward(self, graph):
        # HeteroGraphConv requires an input dictionary, but each relation module
        # owns and uses its source-node embedding matrix.
        dummy_inputs = {
            "user": torch.empty(self.num_users, 0, device=graph.device),
            "item": torch.empty(self.num_items, 0, device=graph.device),
        }
        out_feats = self.conv(graph, dummy_inputs)

        user_out = out_feats["user"]
        item_out = out_feats["item"]

        user_out = self.user_fc(self.dropout(self.activation(user_out)))
        item_out = self.item_fc(self.dropout(self.activation(item_out)))
        return user_out, item_out


class RGCL(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.rgcl_graph = build_rgcl_graph_from_train(cfg)
        self.dgl_graph = self.rgcl_graph["dgl_graph"]

        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)
        self.review_dim = infer_review_dim(cfg)
        self.hidden_dim = int(cfg.model.hidden_dim)
        self.dropout = float(cfg.model.dropout)

        self.lambda_ed = float(cfg.model.lambda_ed)
        self.lambda_nd = float(cfg.model.lambda_nd)

        self.rating_values = torch.tensor(
            self.rgcl_graph["rating_values"],
            dtype=torch.float32,
        )
        self.num_ratings = int(len(self.rgcl_graph["rating_values"]))

        self.encoder = RGCLGraphEncoder(
            num_users=self.num_users,
            num_items=self.num_items,
            review_dim=self.review_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            rating_values=self.rgcl_graph["rating_values"],
        )

        # Same decoder input style as the original ReviewGraph code: [user, item].
        self.pair_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )
        self.rating_predictor = nn.Linear(self.hidden_dim, self.num_ratings, bias=False)
        self.review_proj = nn.Linear(self.review_dim, self.hidden_dim, bias=False)

        self.edge_contrast = ContrastLoss(self.hidden_dim)
        self.node_contrast = ContrastLoss(self.hidden_dim)
        self.rating_loss_fn = nn.MSELoss()

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_graph_device(self, device):
        self.dgl_graph = self.dgl_graph.to(device)
        self.rgcl_graph["dgl_graph"] = self.dgl_graph
        self.rating_values = self.rating_values.to(device)

    def encode(self):
        return self.encoder(self.dgl_graph)

    def make_pair_repr(self, user_emb, item_emb, user_id, item_id):
        user_feat = user_emb[user_id]
        item_feat = item_emb[item_id]
        pair_feat = torch.cat([user_feat, item_feat], dim=-1)
        return self.pair_proj(pair_feat)

    def decode_logits(self, user_emb, item_emb, user_id, item_id):
        edge_h = self.make_pair_repr(user_emb, item_emb, user_id, item_id)
        logits = self.rating_predictor(edge_h)
        return logits, edge_h

    def logits_to_expected_rating(self, logits):
        rating_values = self.rating_values.to(logits.device)
        return (torch.softmax(logits, dim=-1) * rating_values.view(1, -1)).sum(dim=-1)

    def forward(self, user_id, item_id):
        user_emb, item_emb = self.encode()
        logits, _ = self.decode_logits(user_emb, item_emb, user_id, item_id)
        return self.logits_to_expected_rating(logits)

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
            self.edge_contrast(edge_h1, review_h)
            + self.edge_contrast(edge_h2, review_h)
        ) / 2.0

        nd_user_loss = self.node_contrast(user_emb1, user_emb2)
        nd_item_loss = self.node_contrast(item_emb1, item_emb2)
        nd_loss = (nd_user_loss + nd_item_loss) / 2.0

        total_loss = rating_loss + self.lambda_ed * ed_loss + self.lambda_nd * nd_loss

        return {
            "loss": total_loss,
            "rating_loss": rating_loss.detach(),
            "ed_loss": ed_loss.detach(),
            "nd_loss": nd_loss.detach(),
        }
