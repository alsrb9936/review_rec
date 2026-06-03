import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.glove_embedding import build_glove_embedding


class TextCNNEncoder(nn.Module):
    def __init__(self, word_dim: int, num_filters: int, kernel_size: int, output_dim: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=word_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
        )
        self.proj = nn.Linear(num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, doc_emb: torch.Tensor) -> torch.Tensor:
        # doc_emb: [batch_size, seq_len, word_dim]
        if doc_emb.ndim != 3:
            raise ValueError(f"doc_emb must be [batch_size, seq_len, word_dim], got {tuple(doc_emb.shape)}")

        x = doc_emb.float().permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        x = torch.tanh(self.proj(x))
        return self.dropout(x)


class FactorizationMachine(nn.Module):
    def __init__(self, input_dim: int, k: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        self.v = nn.Parameter(torch.randn(input_dim, k) * 0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_part = self.linear(x)
        inter_part1 = torch.mm(x, self.v) ** 2
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = 0.5 * torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True)
        return linear_part + pair_interactions


class TransformNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(dropout))

        for _ in range(max(num_layers - 1, 0)):
            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        word_dim = int(cfg.data.word_dim)
        num_filters = int(cfg.model.num_filters)
        kernel_size = int(cfg.model.kernel_size)
        output_dim = int(cfg.model.output_embedding_size)
        dropout = float(cfg.model.dropout)
        fm_k = int(cfg.model.fm_k)
        num_transform_layers = int(cfg.model.num_transform_layers)
        num_users = int(cfg.stats.num_users)
        num_items = int(cfg.stats.num_items)

        self.rating_loss_weight = float(cfg.model.rating_loss_weight)
        self.act_loss_weight = float(cfg.model.act_loss_weight)
        self.transform_loss_weight = float(cfg.model.transform_loss_weight)
        self.word_embedding = build_glove_embedding(cfg)

        self.user_embedding = nn.Embedding(num_users, output_dim)
        self.item_embedding = nn.Embedding(num_items, output_dim)

        self.user_encoder = TextCNNEncoder(word_dim, num_filters, kernel_size, output_dim, dropout)
        self.item_encoder = TextCNNEncoder(word_dim, num_filters, kernel_size, output_dim, dropout)
        self.review_encoder = TextCNNEncoder(word_dim, num_filters, kernel_size, output_dim, dropout)

        self.transform = TransformNet(
            input_dim=output_dim * 2,
            output_dim=output_dim,
            num_layers=num_transform_layers,
            dropout=dropout,
        )

        # In the original TransNet, the actual-review branch and transformed branch share FM parameters.
        self.shared_fm = FactorizationMachine(output_dim, fm_k)
        self.final_fm = FactorizationMachine(output_dim * 3, fm_k)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss(reduction="sum")

    def encode_source(self, user_doc: torch.Tensor, item_doc: torch.Tensor) -> torch.Tensor:
        user_rep = self.user_encoder(self.word_embedding(user_doc.long()))
        item_rep = self.item_encoder(self.word_embedding(item_doc.long()))
        source = torch.cat([user_rep, item_rep], dim=1)
        return self.transform(source)

    def encode_target(self, target_doc: torch.Tensor) -> torch.Tensor:
        return self.review_encoder(self.word_embedding(target_doc.long()))

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_doc: torch.Tensor,
        item_doc: torch.Tensor,
    ) -> torch.Tensor:
        other_rep = self.encode_source(user_doc, item_doc)
        user_emb = self.user_embedding(user_id.long())
        item_emb = self.item_embedding(item_id.long())
        z = torch.cat([user_emb, item_emb, other_rep], dim=1)
        return self.final_fm(z)

    def act_parameters(self):
        return list(self.review_encoder.parameters()) + list(self.shared_fm.parameters())

    def oth_parameters(self):
        return (
            list(self.user_encoder.parameters())
            + list(self.item_encoder.parameters())
            + list(self.transform.parameters())
        )

    def full_parameters(self):
        return (
            list(self.user_embedding.parameters())
            + list(self.item_embedding.parameters())
            + list(self.final_fm.parameters())
        )

    def train_step(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_doc: torch.Tensor,
        item_doc: torch.Tensor,
        target_doc: torch.Tensor,
        rating: torch.Tensor,
        optimizers: dict[str, torch.optim.Optimizer],
    ) -> dict[str, torch.Tensor]:
        rating = rating.view(-1, 1).float()

        optimizers["act"].zero_grad()
        rev_rep = self.encode_target(target_doc)
        act_pred = self.shared_fm(rev_rep)
        act_loss = self.l1_loss(act_pred, rating) * self.act_loss_weight
        act_loss.backward()
        optimizers["act"].step()

        optimizers["oth"].zero_grad()
        other_rep = self.encode_source(user_doc, item_doc)
        with torch.no_grad():
            rev_target = self.encode_target(target_doc)
        oth_loss = self.mse_loss(other_rep, rev_target.detach()) * self.transform_loss_weight
        oth_loss.backward()
        optimizers["oth"].step()

        with torch.no_grad():
            oth_pred = self.shared_fm(other_rep.detach())
            oth_rating_loss = self.mse_loss(oth_pred, rating)

        optimizers["full"].zero_grad()
        with torch.no_grad():
            fixed_other_rep = self.encode_source(user_doc, item_doc)
        user_emb = self.user_embedding(user_id.long())
        item_emb = self.item_embedding(item_id.long())
        z = torch.cat([user_emb, item_emb, fixed_other_rep.detach()], dim=1)
        full_pred = self.final_fm(z)
        full_loss = self.l1_loss(full_pred, rating) * self.rating_loss_weight
        full_loss.backward()
        optimizers["full"].step()

        return {
            "loss": act_loss.detach() + oth_loss.detach() + full_loss.detach(),
            "act_loss": act_loss.detach(),
            "oth_loss": oth_loss.detach(),
            "full_loss": full_loss.detach(),
            "oth_rating_loss": oth_rating_loss.detach(),
        }

    def calculate_loss(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_doc: torch.Tensor,
        item_doc: torch.Tensor,
        target_doc: torch.Tensor,
        rating: torch.Tensor,
    ) -> torch.Tensor:
        rating = rating.view(-1, 1).float()

        other_rep = self.encode_source(user_doc, item_doc)
        rev_rep = self.encode_target(target_doc)

        user_emb = self.user_embedding(user_id.long())
        item_emb = self.item_embedding(item_id.long())
        z = torch.cat([user_emb, item_emb, other_rep], dim=1)
        pred = self.final_fm(z)
        act_pred = self.shared_fm(rev_rep)

        rating_loss = self.l1_loss(pred, rating)
        act_loss = self.l1_loss(act_pred, rating)
        transform_loss = self.mse_loss(other_rep, rev_rep)

        return (
            self.rating_loss_weight * rating_loss
            + self.act_loss_weight * act_loss
            + self.transform_loss_weight * transform_loss
        )
