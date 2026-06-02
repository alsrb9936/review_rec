import torch
import torch.nn as nn
from omegaconf import DictConfig


class LightGCNEncoder(nn.Module):
    """Pure LightGCN encoder with trainable user/item ID embeddings.

    No review text or review embedding is consumed here. The normalized adjacency
    matrix is built from train interactions and passed once at model construction.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        norm_adj: torch.Tensor,
        embedding_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.register_buffer("norm_adj", norm_adj.coalesce(), persistent=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def get_all_embeddings(self):
        embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0,
        )
        layer_outputs = [embeddings]

        for _ in range(self.num_layers):
            embeddings = torch.sparse.mm(self.norm_adj, embeddings)
            layer_outputs.append(embeddings)

        final_embeddings = torch.stack(layer_outputs, dim=0).mean(dim=0)
        user_embeddings, item_embeddings = torch.split(
            final_embeddings,
            [self.num_users, self.num_items],
            dim=0,
        )
        return user_embeddings, item_embeddings

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor):
        user_all, item_all = self.get_all_embeddings()
        return user_all[user_id], item_all[item_id]


class LightGCN(nn.Module):
    def __init__(self, cfg: DictConfig, norm_adj: torch.Tensor):
        super().__init__()
        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)
        self.embedding_dim = int(cfg.model.embedding_dim)
        self.num_layers = int(cfg.model.num_layers)

        self.graph_encoder = LightGCNEncoder(
            self.num_users,
            self.num_items,
            norm_adj,
            self.embedding_dim,
            self.num_layers,
        )
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.loss_fn = nn.MSELoss()
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor):
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)

        user_emb, item_emb = self.graph_encoder(user_id, item_id)
        rating_pred = torch.sum(user_emb * item_emb, dim=-1)
        rating_pred = rating_pred + self.user_bias(user_id).squeeze(-1)
        rating_pred = rating_pred + self.item_bias(item_id).squeeze(-1)
        return rating_pred + self.global_bias

    def calculate_loss(self, user_id: torch.Tensor, item_id: torch.Tensor, rating: torch.Tensor):
        prediction = self.forward(user_id=user_id, item_id=item_id)
        return self.loss_fn(prediction, rating.view(-1).float())
