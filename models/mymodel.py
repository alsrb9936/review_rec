import torch
import torch.nn as nn
from omegaconf import DictConfig
from models.base_model import BaseModel

class RatingGraphEncoder(nn.Module):
    def __init__(self, num_users, num_items, d_id=64, d_model=128, num_layers=2):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.num_nodes = self.num_users + self.num_items
        self.d_id = int(d_id)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.user_embedding = nn.Embedding(self.num_users, self.d_id)
        self.item_embedding = nn.Embedding(self.num_items, self.d_id)
        self.projection = nn.Linear(self.d_id * 3, self.d_model)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def _propagate_once(self, node_embeddings, edge_index, edge_weight=None):
        src = edge_index[0]
        dst = edge_index[1]
        if edge_weight is None:
            base_weight = node_embeddings.new_ones(src.size(0))
        else:
            base_weight = edge_weight

        degree = node_embeddings.new_zeros(self.num_nodes)
        degree.index_add_(0, src, base_weight)
        norm = base_weight / torch.sqrt(degree[src].clamp_min(1e-8) * degree[dst].clamp_min(1e-8))

        aggregated = node_embeddings.new_zeros(node_embeddings.size())
        aggregated.index_add_(0, dst, node_embeddings[src] * norm.unsqueeze(-1))
        return aggregated

    def _compute_node_embeddings(self, edge_index, edge_weight=None):
        edge_index = edge_index.to(self.user_embedding.weight.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.user_embedding.weight.device)

        initial_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_outputs = [initial_embeddings]
        propagated = initial_embeddings
        for _ in range(self.num_layers):
            propagated = self._propagate_once(propagated, edge_index=edge_index, edge_weight=edge_weight)
            layer_outputs.append(propagated)
        stacked = torch.stack(layer_outputs, dim=0)
        return stacked.mean(dim=0)

    def forward(self, user_ids, item_ids, edge_index, edge_weight=None):
        all_embeddings = self._compute_node_embeddings(edge_index=edge_index, edge_weight=edge_weight)
        user_embeddings = all_embeddings[user_ids]
        item_embeddings = all_embeddings[self.num_users + item_ids]
        interaction_embeddings = torch.cat(
            [user_embeddings, item_embeddings, user_embeddings * item_embeddings],
            dim=-1,
        )
        return self.projection(interaction_embeddings)

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

class MyModel(BaseModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.num_users = cfg.stats.num_users
        self.num_items = cfg.stats.num_items
        self.d_id = cfg.model.d_id
        self.d_model = cfg.model.d_model
        self.d_text = cfg.model.d_text
        self.dropout = cfg.model.dropout
        
        self.loss_fn = nn.MSELoss()
        self._init_weights()

    def _init_weights(self):
        for module in self.mlp_layers_user:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.mlp_layers_item:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_normal_(self.user_mf_embedding.weight)
        nn.init.xavier_normal_(self.item_mf_embedding.weight)

        nn.init.xavier_normal_(self.predict_layer.weight)
        if self.predict_layer.bias is not None:
            nn.init.zeros_(self.predict_layer.bias)

    def forward(self, user_id, item_id, user_review, item_review):
        user_mf_e = self.embedding_dropout(self.user_mf_embedding(user_id))
        item_mf_e = self.embedding_dropout(self.item_mf_embedding(item_id))
        mf_output = user_mf_e * item_mf_e

        mlp_output_user = self.mlp_layers_user(user_review)
        mlp_output_item = self.mlp_layers_item(item_review)

        final_input = torch.cat([mf_output, mlp_output_user, mlp_output_item], dim=-1)
        rating_pred = self.predict_layer(final_input).squeeze(-1)
        return rating_pred

    def calculate_loss(self, user_id, item_id, user_review, item_review, rating):
        prediction = self.forward(user_id, item_id, user_review, item_review)
        return self.loss_fn(prediction, rating)
