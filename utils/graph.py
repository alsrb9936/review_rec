import numpy as np
import scipy.sparse as sp
import torch


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