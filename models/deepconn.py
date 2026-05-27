import numpy as np
import torch
import torch.nn as nn
from .base_model import BaseModel, TextCNN, TorchFM

class DeepCoNN(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        embedding_matrix = np.load(cfg.data.word_embedding_path)
        embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

        self.word_embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=bool(cfg.model.freeze_word_embedding),
            padding_idx=int(cfg.data.pad_id),
        )
        
        word_embed_size = embedding_matrix.shape[1]
        hidden_dim = int(cfg.model.hidden_dim)
        num_filters = int(cfg.model.num_filters)
        kernel_size = int(cfg.model.kernel_size)
        dropout = float(cfg.model.dropout)

        hyper_params = {
            "word_embed_size": word_embed_size,
            "latent_size": hidden_dim,
            "num_filters": num_filters,
            "kernel_size": kernel_size,
            "dropout": dropout,
        }

        self.user_cnn = TextCNN(hyper_params)
        self.item_cnn = TextCNN(hyper_params)
        self.global_bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.fm = TorchFM(n=hidden_dim * 2, k=int(cfg.model.fm_k))
        self.loss_fn = nn.MSELoss()


    def forward(self, user_reviews, item_reviews):
        user_emb = self.word_embedding(user_reviews)
        item_emb = self.word_embedding(item_reviews)

        batch_size = user_emb.size(0)

        user_emb = user_emb.view(batch_size, -1, user_emb.size(-1))
        item_emb = item_emb.view(batch_size, -1, item_emb.size(-1))

        user_vec = self.user_cnn(user_emb)
        item_vec = self.item_cnn(item_emb)

        x = torch.cat([user_vec, item_vec], dim=-1)
        prediction = self.fm(x).squeeze(-1) + self.global_bias

        return prediction

    def calculate_loss(self, user_reviews, item_reviews, rating):
        prediction = self.forward(user_reviews, item_reviews)
        return self.loss_fn(prediction, rating.view(-1).float())