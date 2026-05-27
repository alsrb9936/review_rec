import numpy as np
import torch
import torch.nn as nn
from .base_model import BaseModel, TextCNN, TorchFM

class DeepCoNN(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.user_text_cnn = TextCNN(cfg.model.hyper_params)
        self.item_text_cnn = TextCNN(cfg.model.hyper_params)

        self.fm = TorchFM(n=cfg.data.num_users + cfg.data.num_items, k=cfg.model.hyper_params['latent_size'] * 2)

    def forward(self, user_reviews, item_reviews):
        user_latent = self.user_text_cnn(user_reviews)
        item_latent = self.item_text_cnn(item_reviews)

        # Concatenate user and item latent vectors
        x = torch.cat([user_latent, item_latent], dim=1)  # [B, latent_size * 2]

        # Pass through FM layer
        output = self.fm(x)  # [B, 1]

        return output.squeeze()