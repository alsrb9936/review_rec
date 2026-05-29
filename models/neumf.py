import torch
import torch.nn as nn
from omegaconf import DictConfig
from models.base_model import BaseModel


class NeuMF(BaseModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        num_users = cfg.stats.num_users
        num_items = cfg.stats.num_items
        mf_embedding_size = cfg.model.mf_embedding_size
        mlp_embedding_size = cfg.model.mlp_embedding_size
        mlp_hidden_size = list(cfg.model.mlp_hidden_size)

        self.user_mf_embedding = nn.Embedding(num_users, mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(num_items, mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(num_users, mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(num_items, mlp_embedding_size)

        self.embedding_dropout = nn.Dropout(p=cfg.model.dropout)

        mlp_layers = []
        input_dim = mlp_embedding_size * 2
        for hidden_dim in mlp_hidden_size:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=cfg.model.dropout))
            input_dim = hidden_dim
        self.mlp_layers = nn.Sequential(*mlp_layers)

        mlp_output_dim = input_dim
        self.predict_layer = nn.Linear(mf_embedding_size + mlp_output_dim, 1)

        self.loss_fn = nn.MSELoss()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_mf_embedding.weight)
        nn.init.xavier_normal_(self.item_mf_embedding.weight)
        nn.init.xavier_normal_(self.user_mlp_embedding.weight)
        nn.init.xavier_normal_(self.item_mlp_embedding.weight)

        for module in self.mlp_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_normal_(self.predict_layer.weight)
        if self.predict_layer.bias is not None:
            nn.init.zeros_(self.predict_layer.bias)

    def forward(self, user_id, item_id):
        user_mf_e = self.embedding_dropout(self.user_mf_embedding(user_id))
        item_mf_e = self.embedding_dropout(self.item_mf_embedding(item_id))
        mf_output = user_mf_e * item_mf_e

        user_mlp_e = self.embedding_dropout(self.user_mlp_embedding(user_id))
        item_mlp_e = self.embedding_dropout(self.item_mlp_embedding(item_id))
        mlp_input = torch.cat([user_mlp_e, item_mlp_e], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        final_input = torch.cat([mf_output, mlp_output], dim=-1)
        rating_pred = self.predict_layer(final_input).squeeze(-1)
        return rating_pred

    def calculate_loss(self, user_id, item_id, rating):
        prediction = self.forward(user_id, item_id)
        return self.loss_fn(prediction, rating)
