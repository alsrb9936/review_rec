import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import Optional


class ReviewEncoder(torch.nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        preference_id_count: int,
        quality_id_count: int,
        quality_padding_idx: Optional[int] = None,
    ):
        super().__init__()

        self.preference_id_embedding = torch.nn.Embedding(
            preference_id_count,
            cfg.model.hidden_dim,
        )

        self.quality_id_embedding = torch.nn.Embedding(
            quality_id_count,
            cfg.model.hidden_dim,
            padding_idx=quality_padding_idx,
        )
        self.num_filters = cfg.model.num_filters
        self.review_count = cfg.data.review_count
        self.review_length = cfg.data.review_length

        self.hidden_dim = cfg.model.hidden_dim
        self.kernel_size = cfg.model.kernel_size
        self.dropout = cfg.model.dropout
        self.word_dim = cfg.data.word_dim

        self.conv = torch.nn.Conv1d(
            in_channels=self.word_dim,
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=1,
        )
        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=self.review_length - self.kernel_size + 1,
            stride=1,
        )

        self.att_review = torch.nn.Linear(self.num_filters, self.hidden_dim)
        self.att_id = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.att_layer = torch.nn.Linear(self.hidden_dim, 1)

        self.top_linear = torch.nn.Linear(self.num_filters, self.hidden_dim)
        self.dropout = torch.nn.Dropout(self.dropout)
        self.lossfn = torch.nn.MSELoss()

    def forward(self, review_emb, preference_id, quality_id):
        preference_id = preference_id.view(-1)

        preference_id_emb = self.preference_id_embedding(preference_id)
        quality_id_emb = self.quality_id_embedding(quality_id)

        batch_size = review_emb.shape[0]
        review_in_one = review_emb.view(-1, self.review_length, self.word_dim)
        review_in_one = review_in_one.permute(0, 2, 1)
        review_conv = F.relu(self.conv(review_in_one))
        review_conv = self.max_pool(review_conv).view(-1, self.num_filters)
        review_in_many = review_conv.view(batch_size, self.review_count, -1)

        review_att = self.att_review(review_in_many)
        id_att = self.att_id(quality_id_emb)
        att_weight = self.att_layer(F.relu(review_att + id_att))
        att_weight = F.softmax(att_weight, dim=1)
        att_out = (att_weight * review_in_many).sum(1)

        feature = self.dropout(att_out)
        feature = self.top_linear(feature)
        feature = preference_id_emb + feature
        return feature


class LatentFactor(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.linear = torch.nn.Linear(cfg.model.hidden_dim, 1)
        self.b_user = torch.nn.Parameter(torch.randn([cfg.stats.num_users]), requires_grad=True)
        self.b_item = torch.nn.Parameter(torch.randn([cfg.stats.num_items]), requires_grad=True)

    def forward(self, user_feature, user_id, item_feature, item_id):
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)

        dot = user_feature * item_feature
        predict = (
            self.linear(dot)
            + self.b_user[user_id].view(-1, 1)
            + self.b_item[item_id].view(-1, 1)
        )
        return predict


class NARRE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_users = cfg.stats.num_users
        self.num_items = cfg.stats.num_items
        self.pad_user_id = self.num_users
        self.pad_item_id = self.num_items

        self.user_review_layer = ReviewEncoder(
            cfg,
            preference_id_count=self.num_users,
            quality_id_count=self.num_items + 1,
            quality_padding_idx=self.pad_item_id,
        )

        self.item_review_layer = ReviewEncoder(
            cfg,
            preference_id_count=self.num_items,
            quality_id_count=self.num_users + 1,
            quality_padding_idx=self.pad_user_id,
        )

        self.predict_linear = LatentFactor(cfg)
        self.lossfn = torch.nn.MSELoss()

    def forward(
        self,
        user_id,
        item_id,
        user_review,
        item_review,
        user_review_item_ids,
        item_review_user_ids,
    ):
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)

        user_feature = self.user_review_layer(
            user_review.float(),
            user_id,
            user_review_item_ids,
        )

        item_feature = self.item_review_layer(
            item_review.float(),
            item_id,
            item_review_user_ids,
        )

        predict = self.predict_linear(user_feature, user_id, item_feature, item_id)
        return predict

    def calculate_loss(
        self,
        user_id,
        item_id,
        user_review,
        item_review,
        user_review_item_ids,
        item_review_user_ids,
        rating,
    ):
        prediction = self.forward(
            user_id,
            item_id,
            user_review,
            item_review,
            user_review_item_ids,
            item_review_user_ids,
        )
        return self.lossfn(prediction, rating.view(-1, 1).float())
