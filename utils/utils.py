import os
import random

import numpy as np
import torch
import dgl

from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader



from data import DATASET_DICT


GLOVE_MODEL_NAMES = {"deepconn", "narre", "transnet", "daml", "neumf", "lightgcn"}
BERT_MODEL_NAMES = {"rgcl", "letter", "recafr", "mymodel_v1"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_stats_from_npy(cfg: DictConfig) -> DictConfig:
    if cfg.model_name.lower() in GLOVE_MODEL_NAMES:
        cfg.data.type = "glove"
    elif cfg.model_name.lower() in BERT_MODEL_NAMES:
        cfg.data.type = "bert"

    data_dir = os.path.join(cfg.data.root, cfg.data.dataset, cfg.data.type)
    max_user_id = -1
    max_item_id = -1

    for split in ["train", "valid", "test"]:
        user_path = os.path.join(data_dir, f"{split}_user_id.npy")
        item_path = os.path.join(data_dir, f"{split}_item_id.npy")
        if not os.path.exists(user_path):
            raise FileNotFoundError(f"Missing user id file: {user_path}")
        if not os.path.exists(item_path):
            raise FileNotFoundError(f"Missing item id file: {item_path}")

        user_ids = np.load(user_path)
        item_ids = np.load(item_path)
        if len(user_ids) == 0:
            raise ValueError(f"Empty user id file: {user_path}")
        if len(item_ids) == 0:
            raise ValueError(f"Empty item id file: {item_path}")

        max_user_id = max(max_user_id, int(user_ids.max()))
        max_item_id = max(max_item_id, int(item_ids.max()))

    with open_dict(cfg):
        cfg.stats.num_users = max_user_id + 1
        cfg.stats.num_items = max_item_id + 1

    print(f"num_users: {cfg.stats.num_users}")
    print(f"num_items: {cfg.stats.num_items}")
    return cfg

def build_recafr_norm_adj(cfg) -> torch.Tensor:
    data_type = str(cfg.data.get("type", "bert"))
    if data_type.lower() in {"none", "null", ""}:
        data_type = "bert"
    data_dir = os.path.join(cfg.data.root, cfg.data.dataset, data_type)

    user_path = os.path.join(data_dir, "train_user_id.npy")
    item_path = os.path.join(data_dir, "train_item_id.npy")
    if not os.path.exists(user_path):
        raise FileNotFoundError(f"Missing train user file: {user_path}")
    if not os.path.exists(item_path):
        raise FileNotFoundError(f"Missing train item file: {item_path}")

    user_ids = np.load(user_path).astype(np.int64)
    item_ids = np.load(item_path).astype(np.int64)
    num_users = int(cfg.stats.num_users)
    num_items = int(cfg.stats.num_items)
    num_nodes = num_users + num_items
    item_nodes = item_ids + num_users

    rows = np.concatenate([user_ids, item_nodes])
    cols = np.concatenate([item_nodes, user_ids])
    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes)).coalesce()

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    row, col = adj.indices()
    values = adj.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(adj.indices(), values, size=adj.shape).coalesce()

def build_lightgcn_norm_adj_from_train(cfg: DictConfig) -> torch.Tensor:
    data_dir = os.path.join(cfg.data.root, cfg.data.dataset, cfg.data.type)
    user_path = os.path.join(data_dir, "train_user_id.npy")
    item_path = os.path.join(data_dir, "train_item_id.npy")
    if not os.path.exists(user_path):
        raise FileNotFoundError(f"Missing train user file: {user_path}")
    if not os.path.exists(item_path):
        raise FileNotFoundError(f"Missing train item file: {item_path}")

    user_ids = np.load(user_path).astype(np.int64)
    item_ids = np.load(item_path).astype(np.int64)
    num_users = int(cfg.stats.num_users)
    num_items = int(cfg.stats.num_items)
    num_nodes = num_users + num_items
    item_nodes = item_ids + num_users

    rows = np.concatenate([user_ids, item_nodes])
    cols = np.concatenate([item_nodes, user_ids])
    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes)).coalesce()

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    row, col = adj.indices()
    values = adj.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(adj.indices(), values, size=adj.shape).coalesce()

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


def _node_norm(num_nodes: int, src_ids) -> torch.Tensor:
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

    graph = getattr(dgl, "heterograph")(
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


def get_dataloader(cfg, model_name):
    dataset_cls = DATASET_DICT[model_name]

    if str(model_name).lower() == "rgcl":
        dataset = dataset_cls(cfg, split="train")
        train_datas = getattr(dataset, "train_datas")
        valid_datas = getattr(dataset, "valid_datas")
        test_datas = getattr(dataset, "test_datas")
        print(f"Train size: {len(train_datas[0])}")
        print(f"Valid size: {len(valid_datas[0])}")
        print(f"Test size: {len(test_datas[0])}")
        print("RGCL full-batch graph training enabled")
        return dataset, dataset, dataset

    train_dataset = dataset_cls(cfg, split="train")
    valid_dataset = dataset_cls(cfg, split="valid")
    test_dataset = dataset_cls(cfg, split="test")

    train_batch_size = cfg.training.batch
    train_shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.training.eval_batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.eval_batch, shuffle=False)

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    return train_loader, valid_loader, test_loader
