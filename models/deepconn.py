import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.glove_embedding import build_glove_embedding


class CNN(nn.Module):
    def __init__(self, cfg: DictConfig, word_dim: int):
        super().__init__()

        self.num_filters = int(cfg.model.num_filters)
        self.review_count = int(cfg.data.review_count)
        self.review_length = int(cfg.data.review_length)

        self.hidden_dim = int(cfg.model.hidden_dim)
        self.kernel_size = int(cfg.model.kernel_size)
        self.dropout = float(cfg.model.dropout)

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=word_dim,
                out_channels=self.num_filters,
                kernel_size=self.kernel_size,
                padding=(self.kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.review_length),
            nn.Dropout(p=self.dropout),
        )

        self.linear = nn.Sequential(
            nn.Linear(self.num_filters * self.review_count, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )

    def forward(self, review_emb: torch.Tensor) -> torch.Tensor:
        # review_emb: [batch_size, review_count, review_length, word_dim]
        if review_emb.ndim != 4:
            raise ValueError(
                "CNN expects review_emb shape "
                f"[batch_size, review_count, review_length, word_dim], got {tuple(review_emb.shape)}"
            )

        batch_size, review_count, review_length, word_dim = review_emb.shape
        if review_count != self.review_count:
            raise ValueError(f"review_count mismatch: cfg={self.review_count}, input={review_count}")
        if review_length != self.review_length:
            raise ValueError(f"review_length mismatch: cfg={self.review_length}, input={review_length}")

        review_emb = review_emb.reshape(batch_size * review_count, review_length, word_dim)
        review_emb = review_emb.permute(0, 2, 1)

        latent = self.conv(review_emb)
        latent = latent.reshape(batch_size, self.num_filters * self.review_count)
        latent = self.linear(latent)
        return latent


class FactorizationMachine(nn.Module):
    def __init__(self, p: int, k: int):
        super().__init__()
        self.v = nn.Parameter(torch.rand(p, k) / 10)
        self.linear = nn.Linear(p, 1, bias=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_part = self.linear(x)
        inter_part1 = torch.mm(x, self.v) ** 2
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True)
        pair_interactions = self.dropout(pair_interactions)
        return linear_part + 0.5 * pair_interactions


class DeepCoNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        word_dim = int(cfg.data.word_dim)
        self.word_embedding = build_glove_embedding(cfg)
        self.cnn_u = CNN(cfg, word_dim=word_dim)
        self.cnn_i = CNN(cfg, word_dim=word_dim)
        self.fm = FactorizationMachine(int(cfg.model.hidden_dim) * 2, 10)
        self.lossfn = nn.MSELoss()

    def forward(self, user_review: torch.Tensor, item_review: torch.Tensor) -> torch.Tensor:
        # user_review, item_review: [batch_size, review_count, review_length]
        if user_review.shape != item_review.shape:
            raise ValueError(
                f"user_review and item_review shape mismatch: "
                f"user={tuple(user_review.shape)}, item={tuple(item_review.shape)}"
            )

        user_latent = self.cnn_u(self.word_embedding(user_review.long()))
        item_latent = self.cnn_i(self.word_embedding(item_review.long()))

        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        prediction = self.fm(concat_latent)
        return prediction

    def calculate_loss(self, user_reviews: torch.Tensor, item_reviews: torch.Tensor, rating: torch.Tensor) -> torch.Tensor:
        prediction = self.forward(user_reviews, item_reviews)
        return self.lossfn(prediction, rating.view(-1, 1).float())
