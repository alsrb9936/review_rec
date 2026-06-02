import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class DAML(nn.Module):
    def __init__(self, cfg: DictConfig, word_emb):
        super().__init__()
        self.doc_len = int(cfg.data.doc_len)
        self.filters_num = int(cfg.model.filters_num)
        self.kernel_size = int(cfg.model.kernel_size)
        self.id_emb_size = int(cfg.model.id_emb_size)
        self.dropout_prob = float(cfg.model.dropout_prob)
        self.l2_reg_lambda = float(cfg.model.l2_reg_lambda)
        self.attention_chunk_size = int(cfg.model.attention_chunk_size)

        embedding_weight = torch.as_tensor(word_emb, dtype=torch.float32)
        self.word_dim = int(embedding_weight.size(1))
        configured_word_dim = int(cfg.data.word_dim)
        if configured_word_dim != self.word_dim:
            raise ValueError(
                f"DAML word_dim={configured_word_dim} does not match loaded embedding dim={self.word_dim}"
            )

        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)
        self.pad_id = int(cfg.data.pad_id)

        freeze_word_embedding = bool(cfg.model.freeze_word_embedding)
        self.user_word_embs = nn.Embedding.from_pretrained(
            embedding_weight,
            freeze=freeze_word_embedding,
            padding_idx=self.pad_id,
        )
        self.item_word_embs = nn.Embedding.from_pretrained(
            embedding_weight,
            freeze=freeze_word_embedding,
            padding_idx=self.pad_id,
        )

        word_cnn_padding = (self.kernel_size // 2, 0)
        self.word_cnn = nn.Conv2d(
            1,
            1,
            (self.kernel_size, self.word_dim),
            padding=word_cnn_padding,
        )
        self.user_doc_cnn = nn.Conv2d(
            1,
            self.filters_num,
            (self.kernel_size, self.word_dim),
            padding=word_cnn_padding,
        )
        self.item_doc_cnn = nn.Conv2d(
            1,
            self.filters_num,
            (self.kernel_size, self.word_dim),
            padding=word_cnn_padding,
        )
        self.user_abs_cnn = nn.Conv2d(
            1,
            self.filters_num,
            (self.kernel_size, self.filters_num),
        )
        self.item_abs_cnn = nn.Conv2d(
            1,
            self.filters_num,
            (self.kernel_size, self.filters_num),
        )
        self.unfold = nn.Unfold((3, self.filters_num), padding=(1, 0))

        self.user_fc = nn.Linear(self.filters_num, self.id_emb_size)
        self.item_fc = nn.Linear(self.filters_num, self.id_emb_size)
        self.uid_embedding = nn.Embedding(self.num_users, self.id_emb_size)
        self.iid_embedding = nn.Embedding(self.num_items, self.id_emb_size)
        self.fusion_fc = nn.Linear(self.id_emb_size * 4, 1)
        self.user_bias = nn.Parameter(torch.randn(self.num_users, 1) * 0.01)
        self.item_bias = nn.Parameter(torch.randn(self.num_items, 1) * 0.01)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.lossfn = nn.MSELoss()

        self.reset_parameters()

    def reset_parameters(self):
        for conv in [
            self.word_cnn,
            self.user_doc_cnn,
            self.item_doc_cnn,
            self.user_abs_cnn,
            self.item_abs_cnn,
        ]:
            nn.init.xavier_normal_(conv.weight)
            if conv.bias is not None:
                nn.init.uniform_(conv.bias, -0.1, 0.1)

        for linear in [self.user_fc, self.item_fc, self.fusion_fc]:
            nn.init.uniform_(linear.weight, -0.1, 0.1)
            nn.init.constant_(linear.bias, 0.1)

        nn.init.uniform_(self.uid_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.iid_embedding.weight, -0.1, 0.1)

    def local_attention_cnn(self, word_embs: torch.Tensor, doc_cnn: nn.Conv2d):
        local_att_words = self.word_cnn(word_embs.unsqueeze(1))
        local_word_weight = torch.sigmoid(local_att_words.squeeze(1))
        weighted_word_embs = word_embs * local_word_weight
        return doc_cnn(weighted_word_embs.unsqueeze(1))

    def local_pooling_cnn(
        self,
        feature: torch.Tensor,
        attention: torch.Tensor,
        cnn: nn.Conv2d,
        fc: nn.Linear,
    ):
        batch_size, filters_num, doc_len, _ = feature.shape
        feature = feature.permute(0, 3, 2, 1)
        attention = attention.reshape(batch_size, 1, doc_len, 1)
        pools = feature * attention
        pools = self.unfold(pools)
        pools = pools.reshape(batch_size, 3, filters_num, doc_len)
        pools = pools.sum(dim=1, keepdim=True)
        pools = pools.transpose(2, 3)

        abs_fea = cnn(pools).squeeze(3)
        abs_fea = F.avg_pool1d(abs_fea, abs_fea.size(2))
        return F.relu(fc(abs_fea.squeeze(2)))

    def _chunked_dual_attention(self, user_local_fea: torch.Tensor, item_local_fea: torch.Tensor):
        batch_size, _, doc_len, _ = user_local_fea.shape
        chunk_size = self.attention_chunk_size
        device = user_local_fea.device

        user_attention = torch.zeros(batch_size, doc_len, device=device)
        item_attention = torch.zeros(batch_size, doc_len, device=device)
        item_permuted = item_local_fea.permute(0, 1, 3, 2)

        for start in range(0, doc_len, chunk_size):
            end = min(start + chunk_size, doc_len)
            user_chunk = user_local_fea[:, :, start:end, :]
            diff = user_chunk - item_permuted
            euclidean = diff.pow(2).sum(1).sqrt()
            attn_chunk = 1.0 / (1.0 + euclidean)
            user_attention[:, start:end] = attn_chunk.sum(dim=2)
            item_attention += attn_chunk.sum(dim=1)

        return user_attention, item_attention

    def forward(self, user_id, item_id, user_doc, item_doc):
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)

        user_word_embs = self.user_word_embs(user_doc)
        item_word_embs = self.item_word_embs(item_doc)

        user_local_fea = self.local_attention_cnn(user_word_embs, self.user_doc_cnn)
        item_local_fea = self.local_attention_cnn(item_word_embs, self.item_doc_cnn)

        user_attention, item_attention = self._chunked_dual_attention(user_local_fea, item_local_fea)
        user_doc_fea = self.local_pooling_cnn(
            user_local_fea,
            user_attention,
            self.user_abs_cnn,
            self.user_fc,
        )
        item_doc_fea = self.local_pooling_cnn(
            item_local_fea,
            item_attention,
            self.item_abs_cnn,
            self.item_fc,
        )

        uid_emb = self.uid_embedding(user_id)
        iid_emb = self.iid_embedding(item_id)
        user_feature = torch.stack([user_doc_fea, uid_emb], dim=1).reshape(user_id.size(0), -1)
        item_feature = torch.stack([item_doc_fea, iid_emb], dim=1).reshape(item_id.size(0), -1)
        ui_feature = self.dropout(torch.cat([user_feature, item_feature], dim=1))

        return self.fusion_fc(ui_feature) + self.user_bias[user_id] + self.item_bias[item_id]

    def calculate_loss(self, user_id, item_id, user_doc, item_doc, rating):
        prediction = self.forward(user_id, item_id, user_doc, item_doc)
        rating_loss = self.lossfn(prediction, rating.view(-1, 1).float())

        if self.l2_reg_lambda <= 0:
            return rating_loss

        l2_loss = (
            0.5 * torch.sum(self.fusion_fc.weight ** 2)
            + 0.5 * torch.sum(self.user_fc.weight ** 2)
            + 0.5 * torch.sum(self.item_fc.weight ** 2)
        )
        return rating_loss + self.l2_reg_lambda * l2_loss
