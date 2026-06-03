import torch
import torch.nn as nn
import dgl.function as fn
import dgl.nn.pytorch as dglnn

from utils.utils import rating_to_etype_name


class GCMCGraphConv(nn.Module):
    """Review-aware graph convolution used by the original RGCL code."""

    def __init__(self, num_src_nodes, review_dim, out_feats, dropout_rate=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(num_src_nodes), int(out_feats)))
        self.prob_score = nn.Linear(int(review_dim), 1, bias=False)
        self.review_score = nn.Linear(int(review_dim), 1, bias=False)
        self.review_w = nn.Linear(int(review_dim), int(out_feats), bias=False)
        self.dropout = nn.Dropout(float(dropout_rate))
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
                message = message * self.dropout(edges.src["cj"])
                return {"m": message}

            graph.update_all(message_func, getattr(fn, "sum")(msg="m", out="h"))
            return graph.dstdata["h"] * graph.dstdata["ci"]


class GCMCLayer(nn.Module):
    def __init__(self, rating_vals, num_users, num_movies, review_dim, out_units, dropout_rate=0.0):
        super().__init__()
        self.rating_vals = [float(rating) for rating in rating_vals]
        self.user_fc = nn.Linear(int(out_units), int(out_units))
        self.movie_fc = nn.Linear(int(out_units), int(out_units))
        self.dropout = nn.Dropout(float(dropout_rate))

        sub_conv = {}
        for rating in self.rating_vals:
            etype = rating_to_etype_name(rating)
            sub_conv[etype] = GCMCGraphConv(
                num_src_nodes=num_users,
                review_dim=review_dim,
                out_feats=out_units,
                dropout_rate=dropout_rate,
            )
            sub_conv[f"rev-{etype}"] = GCMCGraphConv(
                num_src_nodes=num_movies,
                review_dim=review_dim,
                out_feats=out_units,
                dropout_rate=dropout_rate,
            )

        self.conv = dglnn.HeteroGraphConv(sub_conv, aggregate="sum")
        self.activation = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, graph, ufeat=None, ifeat=None):
        in_feats = {"user": ufeat, "movie": ifeat}
        out_feats = self.conv(graph, in_feats)
        user_feat = out_feats["user"].view(out_feats["user"].shape[0], -1)
        movie_feat = out_feats["movie"].view(out_feats["movie"].shape[0], -1)

        user_feat = self.user_fc(self.dropout(self.activation(user_feat)))
        movie_feat = self.movie_fc(self.dropout(self.activation(movie_feat)))
        return user_feat, movie_feat


class ContrastLoss(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.w = nn.Parameter(torch.empty(int(feat_size), int(feat_size)))
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        nn.init.xavier_uniform_(self.w)

    def forward(self, x, y, y_neg=None):
        scores = (x @ self.w * y).sum(1)
        pos_loss = self.bce_loss(scores, torch.ones_like(scores))

        if y_neg is None:
            idx = torch.randperm(y.shape[0], device=y.device)
            y_neg = y[idx, :]
        neg_scores = (x @ self.w * y_neg).sum(1)
        neg_loss = self.bce_loss(neg_scores, torch.zeros_like(neg_scores))
        return pos_loss + neg_loss


class MLPPredictorMI(nn.Module):
    def __init__(self, in_units, review_dim, num_classes, dropout_rate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(float(dropout_rate))
        self.review_proj = nn.Linear(int(review_dim), int(in_units), bias=False)
        self.contrast_loss = ContrastLoss(in_units)
        self.linear = nn.Sequential(
            nn.Linear(int(in_units) * 2, int(in_units), bias=False),
            nn.ReLU(),
            nn.Linear(int(in_units), int(in_units), bias=False),
        )
        self.predictor = nn.Linear(int(in_units), int(num_classes), bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def neg_sampling(graph):
        review_feat = graph.edata["review_feat"]
        return review_feat[torch.randperm(review_feat.shape[0], device=review_feat.device), :]

    def apply_edges(self, edges):
        h_u = edges.src["h"]
        h_v = edges.dst["h"]
        h_fea = self.linear(torch.cat([h_u, h_v], dim=1))
        score = self.predictor(h_fea).squeeze()

        if "neg_review_feat" in edges.data:
            review_feat = self.review_proj(edges.data["review_feat"])
            neg_review_feat = self.review_proj(edges.data["neg_review_feat"])
            mi_score = self.contrast_loss(h_fea, review_feat, neg_review_feat)
            return {"score": score, "mi_score": mi_score}
        return {"score": score}

    def forward(self, graph, ufeat, ifeat, cal_edge_mi=True):
        with graph.local_scope():
            graph.nodes["user"].data["h"] = ufeat
            graph.nodes["movie"].data["h"] = ifeat

            if cal_edge_mi and "review_feat" in graph.edata:
                graph.edata["neg_review_feat"] = self.neg_sampling(graph)

            graph.apply_edges(self.apply_edges)
            if "mi_score" in graph.edata:
                return graph.edata["score"], graph.edata["mi_score"]
            return graph.edata["score"]


class RGCL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = None
        self.decoder = None
        self.contrast_loss = None
        self.rating_vals = None
        self.train_classification = bool(cfg.model.get("train_classification", True))
        self.distributed = bool(cfg.model.get("distributed", False))

    def configure_from_dataset(self, dataset):
        rating_vals = dataset.possible_rating_values
        out_units = int(self.cfg.model.hidden_dim)
        dropout = float(self.cfg.model.dropout)
        review_dim = int(dataset.review_fea_size)

        self.rating_vals = torch.tensor(rating_vals, dtype=torch.float32)
        self.encoder = GCMCLayer(
            rating_vals=rating_vals,
            num_users=dataset.num_user,
            num_movies=dataset.num_movie,
            review_dim=review_dim,
            out_units=out_units,
            dropout_rate=dropout,
        )
        num_outputs = len(rating_vals) if self.train_classification else 1
        self.decoder = MLPPredictorMI(
            in_units=out_units,
            review_dim=review_dim,
            num_classes=num_outputs,
            dropout_rate=dropout,
        )
        self.contrast_loss = ContrastLoss(out_units)
        return self

    def forward(self, enc_graph, dec_graph, ufeat=None, ifeat=None, cal_edge_mi=True):
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("RGCL must be configured with configure_from_dataset() before forward().")

        user_out, movie_out = self.encoder(enc_graph, ufeat, ifeat)
        if cal_edge_mi:
            pred_ratings, mi_score = self.decoder(dec_graph, user_out, movie_out, cal_edge_mi=True)
            return pred_ratings, mi_score, user_out, movie_out

        pred_ratings = self.decoder(dec_graph, user_out, movie_out, cal_edge_mi=False)
        return pred_ratings, user_out, movie_out

    def expected_rating(self, pred_ratings):
        if self.rating_vals is None:
            raise RuntimeError("RGCL rating values are not configured.")
        rating_vals = self.rating_vals.to(pred_ratings.device)
        if self.train_classification:
            return (torch.softmax(pred_ratings, dim=1) * rating_vals.view(1, -1)).sum(dim=1)
        return pred_ratings.view(-1)

    @property
    def lambda_ed(self):
        return float(self.cfg.model.lambda_ed)

    @property
    def lambda_nd(self):
        return float(self.cfg.model.lambda_nd)
