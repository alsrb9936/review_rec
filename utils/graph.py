import ast

import numpy as np
import scipy.sparse as sp
import torch


def build_lightgcn_norm_adj(train_df, num_users, num_items):
    users = train_df["user_id"].to_numpy(dtype=np.int64)
    items = train_df["item_id"].to_numpy(dtype=np.int64)
    values = np.ones(len(train_df), dtype=np.float32)

    R = sp.csr_matrix((values, (users, items)), shape=(num_users, num_items), dtype=np.float32)
    zero_u = sp.csr_matrix((num_users, num_users), dtype=np.float32)
    zero_i = sp.csr_matrix((num_items, num_items), dtype=np.float32)
    adj = sp.bmat([[zero_u, R], [R.T, zero_i]], format="csr", dtype=np.float32)

    degree = np.asarray(adj.sum(axis=1)).flatten()
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0

    D_inv_sqrt = sp.diags(degree_inv_sqrt)
    norm_adj = (D_inv_sqrt @ adj @ D_inv_sqrt).tocoo()

    indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def _stack_embeddings(values):
    parsed = []
    for x in values:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            pass
        elif isinstance(x, list):
            x = np.asarray(x, dtype=np.float32)
        elif isinstance(x, str):
            x = np.asarray(ast.literal_eval(x), dtype=np.float32)
        else:
            raise TypeError(f"Unsupported embedding type: {type(x)}")
        parsed.append(np.asarray(x, dtype=np.float32))
    return torch.from_numpy(np.stack(parsed).astype(np.float32))


def _rating_to_etype_name(rating) -> str:
    value = float(rating)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "_")


def _calc_norm(degree: torch.Tensor) -> torch.Tensor:
    degree = degree.float()
    norm = torch.zeros_like(degree, dtype=torch.float32)
    non_zero = degree > 0
    norm[non_zero] = torch.pow(degree[non_zero], -0.5)
    return norm.unsqueeze(1)


def build_rgcl_graph(train_df, num_users, num_items):
    try:
        import dgl
    except ImportError as exc:
        raise ImportError("RGCL graph construction requires DGL.") from exc

    rating_values = sorted(float(r) for r in train_df["rating"].unique().tolist())
    etype_names = [_rating_to_etype_name(r) for r in rating_values]

    users = torch.tensor(train_df["user_id"].values, dtype=torch.long)
    items = torch.tensor(train_df["item_id"].values, dtype=torch.long)
    ratings = torch.tensor(train_df["rating"].values, dtype=torch.float32)
    review_feat = _stack_embeddings(train_df["review_embedding"].to_numpy()).float()

    data_dict = {}
    review_by_etype = {}
    for rating, etype in zip(rating_values, etype_names):
        mask = ratings == float(rating)
        r_users = users[mask]
        r_items = items[mask]
        r_review_feat = review_feat[mask]
        data_dict[("user", etype, "item")] = (r_users, r_items)
        data_dict[("item", f"rev-{etype}", "user")] = (r_items, r_users)
        review_by_etype[etype] = r_review_feat

    graph = dgl.heterograph(data_dict, num_nodes_dict={"user": int(num_users), "item": int(num_items)})

    for etype in etype_names:
        graph[etype].edata["review_feat"] = review_by_etype[etype]
        graph[f"rev-{etype}"].edata["review_feat"] = review_by_etype[etype]

    user_ci_degree = torch.zeros(int(num_users), dtype=torch.float32)
    user_cj_degree = torch.zeros(int(num_users), dtype=torch.float32)
    item_ci_degree = torch.zeros(int(num_items), dtype=torch.float32)
    item_cj_degree = torch.zeros(int(num_items), dtype=torch.float32)

    for etype in etype_names:
        rev_etype = f"rev-{etype}"
        user_ci_degree += graph[rev_etype].in_degrees().float()
        user_cj_degree += graph[etype].out_degrees().float()
        item_ci_degree += graph[etype].in_degrees().float()
        item_cj_degree += graph[rev_etype].out_degrees().float()

    graph.nodes["user"].data["ci"] = _calc_norm(user_ci_degree)
    graph.nodes["user"].data["cj"] = _calc_norm(user_cj_degree)
    graph.nodes["item"].data["ci"] = _calc_norm(item_ci_degree)
    graph.nodes["item"].data["cj"] = _calc_norm(item_cj_degree)

    return {
        "dgl_graph": graph,
        "num_users": int(num_users),
        "num_items": int(num_items),
        "rating_values": rating_values,
        "etype_names": etype_names,
    }
