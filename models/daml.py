import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.glove_embedding import build_glove_embedding


class DAML(nn.Module):
    """KDD 2019 DAML.

    The encoder follows the reference implementation: ``forward(datas)`` consumes
    the DAML tuple layout and returns stacked user/item features with shape
    ``[batch_size, 2, id_emb_size]``.  ``predict`` and ``calculate_loss`` are the
    repo-local adapters used by ``DAMLTrainer`` for rating regression.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.num_fea = 2
        self.doc_len = int(cfg.data.doc_len)
        self.word_dim = int(cfg.data.word_dim)
        self.filters_num = int(cfg.model.filters_num)
        self.kernel_size = int(cfg.model.kernel_size)
        self.id_emb_size = int(cfg.model.id_emb_size)

        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)

        self.user_word_embs = build_glove_embedding(cfg)
        self.item_word_embs = build_glove_embedding(cfg)

        self.word_cnn = nn.Conv2d(1, 1, (5, self.word_dim), padding=(2, 0))
        self.user_doc_cnn = nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim), padding=(1, 0))
        self.item_doc_cnn = nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim), padding=(1, 0))
        self.user_abs_cnn = nn.Conv2d(1, self.filters_num, (self.kernel_size, self.filters_num))
        self.item_abs_cnn = nn.Conv2d(1, self.filters_num, (self.kernel_size, self.filters_num))

        self.unfold = nn.Unfold((3, self.filters_num), padding=(1, 0))

        self.user_fc = nn.Linear(self.filters_num, self.id_emb_size)
        self.item_fc = nn.Linear(self.filters_num, self.id_emb_size)

        self.uid_embedding = nn.Embedding(self.num_users + 2, self.id_emb_size)
        self.iid_embedding = nn.Embedding(self.num_items + 2, self.id_emb_size)

        self.predict_layer = nn.Linear(self.id_emb_size * self.num_fea, 1)
        self.lossfn = nn.MSELoss()

        self.reset_para()

    def forward(self, datas):
        """
        user_reviews, item_reviews, uids, iids, \
        user_item2id, item_user2id, user_doc, item_doc = datas
        """
        _, _, uids, iids, _, _, user_doc, item_doc = datas
        uids = uids.view(-1)
        iids = iids.view(-1)

        user_word_embs = self.user_word_embs(user_doc.long())
        item_word_embs = self.item_word_embs(item_doc.long())

        user_local_fea = self.local_attention_cnn(user_word_embs, self.user_doc_cnn)
        item_local_fea = self.local_attention_cnn(item_word_embs, self.item_doc_cnn)

        euclidean = (user_local_fea - item_local_fea.permute(0, 1, 3, 2)).pow(2).sum(1).sqrt()
        attention_matrix = 1.0 / (1 + euclidean)
        user_attention = attention_matrix.sum(2)
        item_attention = attention_matrix.sum(1)

        user_doc_fea = self.local_pooling_cnn(user_local_fea, user_attention, self.user_abs_cnn, self.user_fc)
        item_doc_fea = self.local_pooling_cnn(item_local_fea, item_attention, self.item_abs_cnn, self.item_fc)

        uid_emb = self.uid_embedding(uids)
        iid_emb = self.iid_embedding(iids)

        user_fea = torch.stack([user_doc_fea, uid_emb], 1)
        item_fea = torch.stack([item_doc_fea, iid_emb], 1)

        return user_fea, item_fea

    def local_attention_cnn(self, word_embs: torch.Tensor, doc_cnn: nn.Conv2d):
        local_att_words = self.word_cnn(word_embs.unsqueeze(1))
        local_word_weight = torch.sigmoid(local_att_words.squeeze(1))
        word_embs = word_embs * local_word_weight
        d_fea = doc_cnn(word_embs.unsqueeze(1))
        return d_fea

    def local_pooling_cnn(self, feature: torch.Tensor, attention: torch.Tensor, cnn: nn.Conv2d, fc: nn.Linear):
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
        abs_fea = F.relu(fc(abs_fea.squeeze(2)))

        return abs_fea

    def predict(self, user_id, item_id, user_doc, item_doc):
        datas = (None, None, user_id, item_id, None, None, user_doc, item_doc)
        user_fea, item_fea = self.forward(datas)
        interaction = (user_fea * item_fea).reshape(user_fea.size(0), -1)
        return self.predict_layer(interaction)

    def calculate_loss(self, user_id, item_id, user_doc, item_doc, rating):
        prediction = self.predict(user_id, item_id, user_doc, item_doc)
        return self.lossfn(prediction, rating.view(-1, 1).float())

    def reset_para(self):
        cnns = [self.word_cnn, self.user_doc_cnn, self.item_doc_cnn, self.user_abs_cnn, self.item_abs_cnn]
        for cnn in cnns:
            nn.init.xavier_normal_(cnn.weight)
            if cnn.bias is not None:
                nn.init.uniform_(cnn.bias, -0.1, 0.1)

        fcs = [self.user_fc, self.item_fc, self.predict_layer]
        for fc in fcs:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

        nn.init.uniform_(self.uid_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.iid_embedding.weight, -0.1, 0.1)
