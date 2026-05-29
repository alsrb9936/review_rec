import torch
import torch.nn as nn
from omegaconf import DictConfig
from models.base_model import BaseModel

class RatingGraphEncoder(nn.Module):
    def __init__(self, num_users, num_items, norm_adj, d_id=64, num_layers=2):
        super().__init__()

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.num_nodes = self.num_users + self.num_items
        self.d_id = int(d_id)
        self.num_layers = int(num_layers)

        self.user_embedding = nn.Embedding(self.num_users, self.d_id)
        self.item_embedding = nn.Embedding(self.num_items, self.d_id)

        self.register_buffer("norm_adj", norm_adj.coalesce())

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_all_embeddings(self):
        x = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0,
        )

        layer_outputs = [x]

        for _ in range(self.num_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            layer_outputs.append(x)

        all_embeddings = torch.stack(layer_outputs, dim=0).mean(dim=0)

        user_embeddings = all_embeddings[:self.num_users]
        item_embeddings = all_embeddings[self.num_users:]

        return user_embeddings, item_embeddings

    def forward(self, user_ids, item_ids):
        user_all, item_all = self.get_all_embeddings()

        user_cf = user_all[user_ids]
        item_cf = item_all[item_ids]

        return user_cf, item_cf

class ReviewProjectionEncoder(nn.Module):
    def __init__(self, input_dim, d_text, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.d_text = d_text
        self.dropout = dropout

        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.d_text),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.layer_norm = nn.LayerNorm(self.d_text)
        self._init_weights()

    def _init_weights(self):
        for module in list(self.projection.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.layer_norm(self.projection(x))

class MyModel(BaseModel):
    def __init__(self, cfg: DictConfig, norm_adj):
        super().__init__(cfg)

        self.num_users = cfg.stats.num_users
        self.num_items = cfg.stats.num_items
        self.d_id = cfg.model.d_id
        self.d_model = cfg.model.d_model
        self.d_text = cfg.model.d_text
        self.dropout = cfg.model.dropout
        self.num_layers = cfg.model.num_layers
        self.input_dim = cfg.data.plm_embedding_size

        self.graph_encoder = RatingGraphEncoder(
            num_users=self.num_users,
            num_items=self.num_items,
            norm_adj=norm_adj,
            d_id=self.d_id,
            num_layers=self.num_layers,
        )

        self.user_review_encoder = ReviewProjectionEncoder(
            input_dim=self.input_dim,
            d_text=self.d_text,
            dropout=self.dropout,
        )

        self.item_review_encoder = ReviewProjectionEncoder(
            input_dim=self.input_dim,
            d_text=self.d_text,
            dropout=self.dropout,
        )
        final_dim = self.d_id * 3 + self.d_text * 2

        self.predict_layer = nn.Sequential(
            nn.Linear(final_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.loss_fn = nn.MSELoss()
        self._init_weights()

    def _init_weights(self):
        # bias terms
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.global_bias)

        # prediction head
        for module in self.predict_layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, user_id, item_id, user_review, item_review):
        user_cf, item_cf = self.graph_encoder(user_id, item_id)

        cf_repr = torch.cat(
            [user_cf, item_cf, user_cf * item_cf],
            dim=-1,
        )

        # user_text = self.user_review_encoder(user_review)
        # item_text = self.item_review_encoder(item_review)

        # final_repr = torch.cat(
        #     [cf_repr, user_text, item_text],
        #     dim=-1,
        # )

        rating_pred = self.predict_layer(cf_repr).squeeze(-1)
        rating_pred = (
            rating_pred
            + self.user_bias(user_id).squeeze(-1)
            + self.item_bias(item_id).squeeze(-1)
            + self.global_bias
        )
        return rating_pred

    def calculate_loss(self, user_id, item_id, user_review, item_review, rating):
        pred = self.forward(user_id, item_id, user_review, item_review)
        return self.loss_fn(pred, rating)
