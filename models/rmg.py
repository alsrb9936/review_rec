import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from models.glove_embedding import build_glove_embedding


class AttentionPool(nn.Module):
    def __init__(self, input_dim: int, att_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, att_dim)
        self.score = nn.Linear(att_dim, 1, bias=False)

    def forward(self, x: torch.Tensor):
        weights = self.score(torch.tanh(self.proj(x))).squeeze(-1)
        weights = F.softmax(weights, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class SentenceEncoder(nn.Module):
    def __init__(self, word_dim: int, cnn_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(word_dim, cnn_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.att = AttentionPool(cnn_dim, cnn_dim)

    def forward(self, word_emb: torch.Tensor):
        # word_emb: [B, sentence_length, word_dim]
        x = self.dropout(word_emb).transpose(1, 2)
        x = F.relu(self.conv(x)).transpose(1, 2)
        x = self.dropout(x)
        return self.att(x)


class DocumentEncoder(nn.Module):
    def __init__(self, word_embedding: nn.Embedding, word_dim: int, cnn_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.word_embedding = word_embedding
        self.sentence_encoder = SentenceEncoder(word_dim, cnn_dim, kernel_size, dropout)
        self.sent_conv = nn.Conv1d(cnn_dim, cnn_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sent_dropout = nn.Dropout(dropout)
        self.sent_att = AttentionPool(cnn_dim, cnn_dim)

    def forward(self, doc_tokens: torch.Tensor):
        # doc_tokens: [B, review_count, sentence_count, sentence_length]
        batch_size, review_count, sentence_count, sentence_length = doc_tokens.shape
        flat_sentences = doc_tokens.reshape(batch_size * review_count * sentence_count, sentence_length)
        word_emb = self.word_embedding(flat_sentences.long())
        sent_emb = self.sentence_encoder(word_emb)
        sent_emb = sent_emb.reshape(batch_size * review_count, sentence_count, -1)
        sent_feat = F.relu(self.sent_conv(sent_emb.transpose(1, 2))).transpose(1, 2)
        sent_feat = self.sent_dropout(sent_feat)
        review_emb = self.sent_att(sent_feat)
        return review_emb.reshape(batch_size, review_count, -1)


class RMG(nn.Module):
    """Review Meets Graph style rating predictor.

    It follows the supplied Keras architecture:
    hierarchical review encoder -> review/document attention -> neighbor graph
    attention -> user/item ID embedding fusion -> elementwise product -> MSE.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)
        self.pad_user_id = self.num_users
        self.pad_item_id = self.num_items
        self.word_dim = int(cfg.data.word_dim)
        self.cnn_dim = int(cfg.model.cnn_dim)
        self.id_dim = int(cfg.model.id_dim)
        self.kernel_size = int(cfg.model.kernel_size)
        self.dropout_prob = float(cfg.model.dropout)

        self.word_embedding = build_glove_embedding(cfg)
        self.doc_encoder = DocumentEncoder(
            self.word_embedding,
            self.word_dim,
            self.cnn_dim,
            self.kernel_size,
            self.dropout_prob,
        )

        self.review_att = AttentionPool(self.cnn_dim, self.cnn_dim)
        self.user_embedding = nn.Embedding(self.num_users + 1, self.id_dim, padding_idx=self.pad_user_id)
        self.item_embedding = nn.Embedding(self.num_items + 1, self.id_dim, padding_idx=self.pad_item_id)

        self.user_neighbor_encoder = AttentionPool(self.id_dim, self.id_dim)
        self.item_neighbor_encoder = AttentionPool(self.id_dim, self.id_dim)
        self.user_graph_att = AttentionPool(self.id_dim * 2, self.cnn_dim)
        self.item_graph_att = AttentionPool(self.id_dim * 2, self.cnn_dim)

        user_factor_dim = self.cnn_dim + self.id_dim + self.id_dim * 2
        item_factor_dim = self.cnn_dim + self.id_dim + self.id_dim * 2
        if user_factor_dim != item_factor_dim:
            raise ValueError("RMG user/item factor dims must match for elementwise product.")
        self.factor_dim = user_factor_dim
        self.prediction = nn.Linear(self.factor_dim, 1)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.lossfn = nn.MSELoss()
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        with torch.no_grad():
            self.user_embedding.weight[self.pad_user_id].zero_()
            self.item_embedding.weight[self.pad_item_id].zero_()

    def _review_factor(self, doc_tokens: torch.Tensor):
        review_emb = self.doc_encoder(doc_tokens)
        return self.review_att(review_emb)

    def _user_graph_factor(self, user_item_ids: torch.Tensor, user_item_user_ids: torch.Tensor):
        # user_item_ids: [B, K], user_item_user_ids: [B, K, K]
        item_neighbor_emb = self.item_embedding(user_item_ids.long())
        second_user_emb = self.user_embedding(user_item_user_ids.long())
        batch_size, neighbor_count, second_count, id_dim = second_user_emb.shape
        encoded_second_users = self.user_neighbor_encoder(
            second_user_emb.reshape(batch_size * neighbor_count, second_count, id_dim)
        ).reshape(batch_size, neighbor_count, id_dim)
        graph_nodes = torch.cat([item_neighbor_emb, encoded_second_users], dim=-1)
        return self.user_graph_att(graph_nodes)

    def _item_graph_factor(self, item_user_ids: torch.Tensor, item_user_item_ids: torch.Tensor):
        # item_user_ids: [B, K], item_user_item_ids: [B, K, K]
        user_neighbor_emb = self.user_embedding(item_user_ids.long())
        second_item_emb = self.item_embedding(item_user_item_ids.long())
        batch_size, neighbor_count, second_count, id_dim = second_item_emb.shape
        encoded_second_items = self.item_neighbor_encoder(
            second_item_emb.reshape(batch_size * neighbor_count, second_count, id_dim)
        ).reshape(batch_size, neighbor_count, id_dim)
        graph_nodes = torch.cat([user_neighbor_emb, encoded_second_items], dim=-1)
        return self.item_graph_att(graph_nodes)

    def forward(
        self,
        user_doc,
        item_doc,
        user_item_user_ids,
        user_item_ids,
        item_user_item_ids,
        item_user_ids,
        item_id,
        user_id,
    ):
        user_id = user_id.view(-1).long()
        item_id = item_id.view(-1).long()
        item_id = torch.clamp(item_id, 0, self.num_items - 1)
        user_id = torch.clamp(user_id, 0, self.num_users - 1)

        user_review_factor = self._review_factor(user_doc)
        item_review_factor = self._review_factor(item_doc)
        user_id_factor = self.user_embedding(user_id)
        item_id_factor = self.item_embedding(item_id)
        user_graph_factor = self._user_graph_factor(user_item_ids, user_item_user_ids)
        item_graph_factor = self._item_graph_factor(item_user_ids, item_user_item_ids)

        user_factor = torch.cat([user_review_factor, user_id_factor, user_graph_factor], dim=-1)
        item_factor = torch.cat([item_review_factor, item_id_factor, item_graph_factor], dim=-1)
        product = self.dropout(user_factor * item_factor)
        return F.relu(self.prediction(product))

    def calculate_loss(
        self,
        user_doc,
        item_doc,
        user_item_user_ids,
        user_item_ids,
        item_user_item_ids,
        item_user_ids,
        item_id,
        user_id,
        rating,
    ):
        prediction = self.forward(
            user_doc=user_doc,
            item_doc=item_doc,
            user_item_user_ids=user_item_user_ids,
            user_item_ids=user_item_ids,
            item_user_item_ids=item_user_item_ids,
            item_user_ids=item_user_ids,
            item_id=item_id,
            user_id=user_id,
        )
        return self.lossfn(prediction, rating.view(-1, 1).float())
