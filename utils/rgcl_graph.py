# utils/rgcl_graph.py
import os

import dgl
import numpy as np
import torch


def rating_to_etype_name(rating) -> str:
    value = float(rating)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "_")


def get_rgcl_data_dir(cfg):
    data_type = str(cfg.data.get("type", "bert"))
    if data_type.lower() in {"none", "null", ""}:
        data_type = "bert"

    return os.path.join(
        cfg.data.root,
        cfg.data.dataset,
        data_type,
    )


def infer_review_dim(cfg) -> int:
    review_path = os.path.join(get_rgcl_data_dir(cfg), "review_emb.npy")
    if not os.path.exists(review_path):
        raise FileNotFoundError(f"Missing review embedding file: {review_path}")
    return int(np.load(review_path, mmap_mode="r").shape[1])


def _load_train_arrays(cfg):
    data_dir = get_rgcl_data_dir(cfg)

    user_ids = np.load(os.path.join(data_dir, "train_user_id.npy")).astype(np.int64)
    item_ids = np.load(os.path.join(data_dir, "train_item_id.npy")).astype(np.int64)
    ratings = np.load(os.path.join(data_dir, "train_rating.npy")).astype(np.float32)
    review_feat = np.load(os.path.join(data_dir, "review_emb.npy")).astype(np.float32)

    if not (len(user_ids) == len(item_ids) == len(ratings) == len(review_feat)):
        raise ValueError(
            "RGCL train arrays must have the same first dimension: "
            f"users={len(user_ids)}, items={len(item_ids)}, "
            f"ratings={len(ratings)}, reviews={len(review_feat)}"
        )

    return user_ids, item_ids, ratings, review_feat


def _node_norm(num_nodes: int, src_ids: np.ndarray) -> torch.Tensor:
    degree = np.bincount(src_ids, minlength=num_nodes).astype(np.float32)
    degree[degree == 0.0] = 1.0
    norm = 1.0 / np.sqrt(degree)
    return torch.from_numpy(norm).float().unsqueeze(-1)


def build_rgcl_graph_from_train(cfg):
    """Build the train-only heterogeneous graph required by RGCL.

    Graph schema:
        user -rating-> item
        item -rev-rating-> user

    Edge data:
        review_feat: train review embedding aligned with each train interaction.

    Node data:
        ci, cj: degree normalization coefficients used by ReviewAwareGraphConv.
    """

    user_ids, item_ids, ratings, review_feat = _load_train_arrays(cfg)

    num_users = int(cfg.stats.num_users)
    num_items = int(cfg.stats.num_items)
    rating_values = sorted(np.unique(ratings).astype(np.float32).tolist())

    data_dict = {}
    review_dict = {}

    for rating in rating_values:
        etype = rating_to_etype_name(rating)
        mask = ratings == rating

        rating_user_ids = user_ids[mask]
        rating_item_ids = item_ids[mask]
        rating_review_feat = review_feat[mask]

        forward_etype = ("user", etype, "item")
        reverse_etype = ("item", f"rev-{etype}", "user")

        data_dict[forward_etype] = (
            torch.from_numpy(rating_user_ids).long(),
            torch.from_numpy(rating_item_ids).long(),
        )
        data_dict[reverse_etype] = (
            torch.from_numpy(rating_item_ids).long(),
            torch.from_numpy(rating_user_ids).long(),
        )

        feat = torch.from_numpy(rating_review_feat).float()
        review_dict[forward_etype] = feat
        review_dict[reverse_etype] = feat

    graph = dgl.heterograph(
        data_dict,
        num_nodes_dict={
            "user": num_users,
            "item": num_items,
        },
    )

    user_norm = _node_norm(num_users, user_ids)
    item_norm = _node_norm(num_items, item_ids)

    graph.nodes["user"].data["ci"] = user_norm
    graph.nodes["user"].data["cj"] = user_norm
    graph.nodes["item"].data["ci"] = item_norm
    graph.nodes["item"].data["cj"] = item_norm

    for etype, feat in review_dict.items():
        graph.edges[etype].data["review_feat"] = feat

    return {
        "dgl_graph": graph,
        "rating_values": rating_values,
    }
