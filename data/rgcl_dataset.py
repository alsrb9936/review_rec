import os

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

def _get_data_dir(cfg):
    data_type = str(cfg.data.get("type", "bert"))
    if data_type.lower() in {"none", "null", ""}:
        data_type = "bert"
    return os.path.join(cfg.data.root, cfg.data.dataset, data_type)


def rating_to_etype_name(rating):
    value = float(rating)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "_")


class RGCLDataset(Dataset[object]):
    def __init__(self, cfg, split: str = "train"):
        self.cfg = cfg
        self.data_dir = _get_data_dir(cfg)
        self._device = torch.device("cpu")

        self._num_user = int(cfg.stats.num_users)
        self._num_movie = int(cfg.stats.num_items)

        self.train_datas = self._load_split("train")
        self.valid_datas = self._load_split("valid")
        self.test_datas = self._load_split("test")

        review_path = os.path.join(self.data_dir, "review_emb.npy")
        if not os.path.exists(review_path):
            raise FileNotFoundError(f"Missing train review embedding file: {review_path}")
        train_review_matrix = np.load(review_path).astype(np.float32)
        if len(train_review_matrix) != len(self.train_datas[0]):
            raise ValueError(
                "review_emb.npy must align with train interactions: "
                f"review_emb={len(train_review_matrix)}, train={len(self.train_datas[0])}"
            )

        self.review_fea_size = int(train_review_matrix.shape[1])
        self.train_review_matrix = torch.from_numpy(train_review_matrix).float()
        self.train_review_feat = self._make_train_review_dict(self.train_review_matrix)

        self.possible_rating_values = np.unique(self.train_datas[2]).astype(np.float32)

        self.user_feature = None
        self.movie_feature = None
        self.user_feature_shape = (self.num_user, self.num_user)
        self.movie_feature_shape = (self.num_movie, self.num_movie)

        train_rating_pairs, train_rating_values = self._generate_pair_value("train")
        valid_rating_pairs, valid_rating_values = self._generate_pair_value("valid")
        test_rating_pairs, test_rating_values = self._generate_pair_value("test")

        self.train_enc_graph = self._generate_enc_graph(train_rating_pairs, train_rating_values)
        self.train_dec_graph = self._generate_dec_graph(
            train_rating_pairs,
            review_feat=self.train_review_feat,
        )
        self.train_labels = self._make_labels(train_rating_values)
        self.train_truths = torch.from_numpy(train_rating_values).float()

        self.valid_enc_graph = self.train_enc_graph
        self.valid_dec_graph = self._generate_dec_graph(valid_rating_pairs)
        self.valid_labels = self._make_labels(valid_rating_values)
        self.valid_truths = torch.from_numpy(valid_rating_values).float()

        self.test_enc_graph = self.train_enc_graph
        self.test_dec_graph = self._generate_dec_graph(test_rating_pairs)
        self.test_labels = self._make_labels(test_rating_values)
        self.test_truths = torch.from_numpy(test_rating_values).float()

    def _load_split(self, split):
        user_ids = np.load(os.path.join(self.data_dir, f"{split}_user_id.npy")).astype(np.int64)
        item_ids = np.load(os.path.join(self.data_dir, f"{split}_item_id.npy")).astype(np.int64)
        ratings = np.load(os.path.join(self.data_dir, f"{split}_rating.npy")).astype(np.float32)
        if not (len(user_ids) == len(item_ids) == len(ratings)):
            raise ValueError(
                f"Length mismatch in {split}: "
                f"users={len(user_ids)}, items={len(item_ids)}, ratings={len(ratings)}"
            )
        return user_ids.tolist(), item_ids.tolist(), ratings.tolist()

    def _make_train_review_dict(self, review_matrix):
        user_ids, item_ids, _ = self.train_datas
        return {
            (int(user_id), int(item_id)): review_matrix[idx]
            for idx, (user_id, item_id) in enumerate(zip(user_ids, item_ids))
        }

    def _make_labels(self, ratings):
        return torch.LongTensor(np.searchsorted(self.possible_rating_values, ratings))

    def _generate_pair_value(self, sub_dataset):
        if sub_dataset == "train":
            user_id, item_id, rating = self.train_datas
        elif sub_dataset == "valid":
            user_id, item_id, rating = self.valid_datas
        elif sub_dataset == "test":
            user_id, item_id, rating = self.test_datas
        else:
            raise ValueError(f"Unsupported split: {sub_dataset}")

        rating_pairs = (
            np.array(user_id, dtype=np.int64),
            np.array(item_id, dtype=np.int64),
        )
        rating_values = np.array(rating, dtype=np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values):
        rating_row, rating_col = rating_pairs
        data_dict = {}
        review_data_dict = {}

        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)[0]
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            etype = rating_to_etype_name(rating)

            data_dict[("user", etype, "movie")] = (rrow, rcol)
            data_dict[("movie", f"rev-{etype}", "user")] = (rcol, rrow)

            review_data_dict[etype] = torch.stack(
                [self.train_review_feat[(int(rating_row[idx]), int(rating_col[idx]))] for idx in ridx]
            ).float()

        graph = getattr(dgl, "heterograph")(
            data_dict,
            num_nodes_dict={"user": self.num_user, "movie": self.num_movie},
        )

        for rating in self.possible_rating_values:
            etype = rating_to_etype_name(rating)
            graph.edges[etype].data["review_feat"] = review_data_dict[etype]
            graph.edges[f"rev-{etype}"].data["review_feat"] = review_data_dict[etype]

        self._add_support(graph)
        return graph

    def _add_support(self, graph):
        def calc_norm(x):
            x = x.numpy().astype("float32")
            x[x == 0.0] = np.inf
            return torch.FloatTensor(1.0 / np.sqrt(x)).unsqueeze(1)

        user_ci = []
        user_cj = []
        movie_ci = []
        movie_cj = []

        for rating in self.possible_rating_values:
            etype = rating_to_etype_name(rating)
            user_ci.append(graph[f"rev-{etype}"].in_degrees())
            movie_ci.append(graph[etype].in_degrees())
            user_cj.append(graph[etype].out_degrees())
            movie_cj.append(graph[f"rev-{etype}"].out_degrees())

        graph.nodes["user"].data["ci"] = calc_norm(sum(user_ci))
        graph.nodes["user"].data["cj"] = calc_norm(sum(user_cj))
        graph.nodes["movie"].data["ci"] = calc_norm(sum(movie_ci))
        graph.nodes["movie"].data["cj"] = calc_norm(sum(movie_cj))

    def _generate_dec_graph(self, rating_pairs, review_feat=None):
        ones = np.ones_like(rating_pairs[0])
        user_movie_ratings = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_user, self.num_movie),
            dtype=np.float32,
        )
        coo = user_movie_ratings.tocoo()
        graph = getattr(dgl, "heterograph")(
            {("user", "rate", "movie"): (coo.row, coo.col)},
            num_nodes_dict={"user": self.num_user, "movie": self.num_movie},
        )

        if review_feat is not None:
            ui_pairs = list(zip(rating_pairs[0].tolist(), rating_pairs[1].tolist()))
            feat = torch.stack([review_feat[(int(user_id), int(item_id))] for user_id, item_id in ui_pairs])
            graph.edata["review_feat"] = feat.float()

        return graph

    def to(self, device):
        self._device = device
        for name in [
            "train_enc_graph",
            "train_dec_graph",
            "valid_enc_graph",
            "valid_dec_graph",
            "test_enc_graph",
            "test_dec_graph",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        for name in [
            "train_labels",
            "train_truths",
            "valid_labels",
            "valid_truths",
            "test_labels",
            "test_truths",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"dataset": self}

    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie
