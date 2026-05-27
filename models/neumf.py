import torch
import torch.nn as nn
from omegaconf import DictConfig
from models.base_model import BaseModel


class NeuMF(BaseModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        num_users = cfg.stats.num_users
        num_items = cfg.stats.num_items
        embedding_dim = cfg.hidden_dim

        self.user_embedding_mf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mf = nn.Embedding(num_items, embedding_dim)
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        mlp_input_dim = embedding_dim * 2
        mlp_layers = []
        current_dim = mlp_input_dim
        for _ in range(cfg.num_layers - 1):
            next_dim = current_dim // 2
            mlp_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ])
            current_dim = next_dim

        self.mlp = nn.Sequential(*mlp_layers)

        predict_input_dim = embedding_dim + current_dim
        self.predict = nn.Linear(predict_input_dim, 1)

        self.loss_fn = nn.MSELoss()
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, user_id, item_id):
        user_emb_mf = self.user_embedding_mf(user_id)
        item_emb_mf = self.item_embedding_mf(item_id)
        mf_vector = user_emb_mf * item_emb_mf

        user_emb_mlp = self.user_embedding_mlp(user_id)
        item_emb_mlp = self.item_embedding_mlp(item_id)
        mlp_vector = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_vector = self.mlp(mlp_vector)

        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.predict(predict_vector).squeeze(-1)
        return prediction

    def calculate_loss(self, user_id, item_id, rating):
        prediction = self.forward(user_id, item_id)
        return self.loss_fn(prediction, rating)
