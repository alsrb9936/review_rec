import numpy as np
import scipy.sparse as sp
import torch
import ast

def build_lightgcn_norm_adj(train_df, num_users, num_items):
    users = train_df["user_id"].to_numpy(dtype=np.int64)
    items = train_df["item_id"].to_numpy(dtype=np.int64)

    # 기본 LightGCN은 rating 값을 edge weight로 쓰지 않고 interaction 여부만 사용
    values = np.ones(len(train_df), dtype=np.float32)

    R = sp.csr_matrix(
        (values, (users, items)),
        shape=(num_users, num_items),
        dtype=np.float32,
    )

    zero_u = sp.csr_matrix((num_users, num_users), dtype=np.float32)
    zero_i = sp.csr_matrix((num_items, num_items), dtype=np.float32)

    adj = sp.bmat(
        [
            [zero_u, R],
            [R.T, zero_i],
        ],
        format="csr",
        dtype=np.float32,
    )

    degree = np.asarray(adj.sum(axis=1)).flatten()
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0

    D_inv_sqrt = sp.diags(degree_inv_sqrt)
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    norm_adj = norm_adj.tocoo()

    indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)

    return torch.sparse_coo_tensor(indices, values, shape).coalesce()

# utils/rgcl_graph.py
import ast
import numpy as np
import torch


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


def build_rgcl_graph(train_df, num_users, num_items):
    """
    Leakage-free RGCL graph.

    Only train_df is allowed here.
    valid/test rows must never be passed to this function.
    """
    rating_values = sorted(train_df["rating"].unique().tolist())

    users = torch.tensor(train_df["user_id"].values, dtype=torch.long)
    items = torch.tensor(train_df["item_id"].values, dtype=torch.long)
    ratings = torch.tensor(train_df["rating"].values, dtype=torch.float32)
    review_feat = _stack_embeddings(train_df["review_embedding"].to_numpy()).float()

    edge_index_by_rating = {}
    edge_review_by_rating = {}

    for rating in rating_values:
        r = float(rating)
        mask = ratings == r

        edge_index_by_rating[r] = (
            users[mask],
            items[mask],
        )
        edge_review_by_rating[r] = review_feat[mask]

    return {
        "num_users": int(num_users),
        "num_items": int(num_items),
        "rating_values": [float(r) for r in rating_values],
        "users": users,
        "items": items,
        "ratings": ratings,
        "review_feat": review_feat,
        "edge_index_by_rating": edge_index_by_rating,
        "edge_review_by_rating": edge_review_by_rating,
    }