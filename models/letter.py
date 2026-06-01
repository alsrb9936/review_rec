import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from models.base_model import BaseModel


class LetterModel(BaseModel):
    """
    LETTER model aligned with the original released implementation, adapted to
    the project's BaseModel/DictConfig interface.

    Main cleanup from the previous version:
    - no CL / AA / reg auxiliary loss branches
    - calculate_loss() uses only MSE, matching Eq. (6) in the paper
    - forward path returns only rating prediction
    """

    def __init__(self, cfg: DictConfig, letter_data):
        super().__init__(cfg)

        self.num_users = int(cfg.stats.num_users)
        self.num_items = int(cfg.stats.num_items)

        reviews = letter_data["reviews"]
        ratings = letter_data["ratings"]

        self.embedding_dim = int(reviews[0].shape[1])
        self.hidden_dim = int(cfg.model.hidden)
        self.pivot = int(cfg.model.pivot)
        self.edge_ratio = int(cfg.model.edge_ratio)

        device_str = f"cuda:{cfg.experiment.device}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # Frozen review-based node features.
        # reviews order:
        #   0: user all-review embedding
        #   1: item all-review embedding
        #   2: user positive/like-review embedding
        #   3: item positive-review embedding
        #   4: user negative/dislike-review embedding
        #   5: item negative-review embedding
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.user_pos_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.user_neg_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_pos_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.item_neg_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        self.user_embedding.weight = nn.Parameter(torch.from_numpy(reviews[0]).float())
        self.item_embedding.weight = nn.Parameter(torch.from_numpy(reviews[1]).float())
        self.user_pos_embedding.weight = nn.Parameter(torch.from_numpy(reviews[2]).float())
        self.item_pos_embedding.weight = nn.Parameter(torch.from_numpy(reviews[3]).float())
        self.user_neg_embedding.weight = nn.Parameter(torch.from_numpy(reviews[4]).float())
        self.item_neg_embedding.weight = nn.Parameter(torch.from_numpy(reviews[5]).float())

        self.user_embedding.weight.requires_grad = False
        self.user_pos_embedding.weight.requires_grad = False
        self.user_neg_embedding.weight.requires_grad = False
        self.item_embedding.weight.requires_grad = False
        self.item_pos_embedding.weight.requires_grad = False
        self.item_neg_embedding.weight.requires_grad = False

        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)

        # Kept for state/structure parity with the original released code.
        # These two embeddings are initialized but not used in the final prediction path.
        self.user_p = nn.Embedding(self.num_users, self.hidden_dim)
        self.item_p = nn.Embedding(self.num_items, self.hidden_dim)
        nn.init.xavier_uniform_(self.user_p.weight)
        nn.init.xavier_uniform_(self.item_p.weight)

        self.uFC = nn.Sequential(
            nn.Linear(self.num_items, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(),
        )
        self.iFC = nn.Sequential(
            nn.Linear(self.num_users, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(),
        )
        self.ruFC = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(),
        )
        self.riFC = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(),
        )
        self.rupFC = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(),
        )
        self.runFC = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(),
        )
        self.ripFC = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(),
        )
        self.rinFC = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(),
        )
        self.nFC = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(),
        )

        # One-layer GNN modules for G/L/D user reps and item reps.
        self.ugnn1 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.ignn1 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.upnn1 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.ipnn1 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.unnn1 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.innn1 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.ReLU = nn.ReLU()
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim)

        self.user_rating = torch.from_numpy(ratings[0].astype("float32")).to(self.device)
        self.item_rating = torch.from_numpy(ratings[1].astype("float32")).to(self.device)

        # Kept for parity with the released code. These attention modules are unused.
        self.rr_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=8, batch_first=True, dropout=0.3
        )
        self.rr_attn_2 = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=8, batch_first=True, dropout=0.3
        )
        self.rr_attn_up = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=8, batch_first=True, dropout=0.3
        )
        self.rr_attn_un = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=8, batch_first=True, dropout=0.3
        )
        self.rr_attn_ip = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=8, batch_first=True, dropout=0.3
        )
        self.rr_attn_in = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=8, batch_first=True, dropout=0.3
        )

        self.loss_fn = nn.MSELoss()
        self.to(self.device)

    def graph_edge__(self, user_emb, item_emb, user_ids, item_ids):
        k = self.edge_ratio

        norm_u = F.normalize(user_emb, p=2, dim=1)
        norm_i = F.normalize(item_emb, p=2, dim=1)

        num_u = max(1, int(torch.ceil(torch.tensor(k / 100.0 * self.num_users))))
        num_i = max(1, int(torch.ceil(torch.tensor(k / 100.0 * self.num_items))))

        def calculate_mask_in_batches(norm_matrix, num_elements, ids, top_k):
            mask = torch.ones(
                (len(ids), num_elements),
                dtype=torch.bool,
                device=norm_matrix.device,
            )

            # Same operation as the original released code.
            similarities = torch.sparse.mm(norm_matrix[ids], norm_matrix.t())
            probs = nn.Softmax(dim=1)(similarities)

            selected_indices = torch.multinomial(probs, top_k, replacement=False)
            mask[torch.arange(ids.size(0)).unsqueeze(1), selected_indices] = False

            probs = nn.Softmax(dim=1)((~mask) * similarities)
            return mask, probs

        u_mask, u_sims = calculate_mask_in_batches(
            norm_u, self.num_users, user_ids, num_u
        )
        i_mask, i_sims = calculate_mask_in_batches(
            norm_i, self.num_items, item_ids, num_i
        )

        del norm_u, norm_i
        del num_u, num_i

        return u_mask, i_mask, u_sims, i_sims

    def mark_unique_elements(self, tensor):
        unique_elements, inverse_indices, counts = torch.unique(
            tensor, return_inverse=True, return_counts=True
        )

        boolean_tensor = torch.zeros_like(tensor, dtype=torch.bool, device=self.device)

        first_occurrence = torch.zeros_like(
            unique_elements, dtype=torch.bool, device=self.device
        )
        boolean_tensor[first_occurrence[inverse_indices].logical_not()] = True
        first_occurrence[inverse_indices] = True

        return boolean_tensor

    def _predict(self, user_ids, item_ids, clip=0):
        u_mask, i_mask, u_sims, i_sims = self.graph_edge__(
            self.user_rating / 5,
            self.item_rating / 5,
            user_ids,
            item_ids,
        )

        unique_u_mask = (~u_mask).any(dim=0)
        unique_i_mask = (~i_mask).any(dim=0)

        unique_u_mask[user_ids] = True
        unique_i_mask[item_ids] = True

        t_user_ids = torch.searchsorted(
            torch.nonzero(unique_u_mask, as_tuple=False).squeeze(),
            user_ids,
        )
        t_item_ids = torch.searchsorted(
            torch.nonzero(unique_i_mask, as_tuple=False).squeeze(),
            item_ids,
        )

        user_embeds = self.ruFC(self.user_embedding.weight[unique_u_mask])
        item_embeds = self.riFC(self.item_embedding.weight[unique_i_mask])
        user_pos_embeds = self.rupFC(self.user_pos_embedding.weight[unique_u_mask])
        user_neg_embeds = self.runFC(self.user_neg_embedding.weight[unique_u_mask])
        item_pos_embeds = self.ripFC(self.item_pos_embedding.weight[unique_i_mask])
        item_neg_embeds = self.rinFC(self.item_neg_embedding.weight[unique_i_mask])

        user_r_embeds = torch.mm(u_sims[:, unique_u_mask], user_embeds)
        user_r_embeds = self.ugnn1(user_r_embeds)

        item_r_embeds = torch.mm(i_sims[:, unique_i_mask], item_embeds)
        item_r_embeds = self.ignn1(item_r_embeds)

        user_r_embeds = user_r_embeds + user_embeds[t_user_ids]
        item_r_embeds = item_r_embeds + item_embeds[t_item_ids]

        user_pos_r_embeds = torch.mm(u_sims[:, unique_u_mask], user_pos_embeds)
        user_pos_r_embeds = self.upnn1(user_pos_r_embeds)
        user_pos_r_embeds = user_pos_r_embeds + user_pos_embeds[t_user_ids]

        item_pos_r_embeds = torch.mm(i_sims[:, unique_i_mask], item_pos_embeds)
        item_pos_r_embeds = self.ipnn1(item_pos_r_embeds)
        item_pos_r_embeds = item_pos_r_embeds + item_pos_embeds[t_item_ids]

        user_neg_r_embeds = torch.mm(u_sims[:, unique_u_mask], user_neg_embeds)
        user_neg_r_embeds = self.unnn1(user_neg_r_embeds)
        user_neg_r_embeds = user_neg_r_embeds + user_neg_embeds[t_user_ids]

        item_neg_r_embeds = torch.mm(i_sims[:, unique_i_mask], item_neg_embeds)
        item_neg_r_embeds = self.innn1(item_neg_r_embeds)
        item_neg_r_embeds = item_neg_r_embeds + item_neg_embeds[t_item_ids]

        user_biases = self.user_bias(user_ids).squeeze()
        item_biases = self.item_bias(item_ids).squeeze()

        dot_product = 0
        dot_product = dot_product + (user_r_embeds * item_r_embeds).sum(1)
        dot_product = dot_product + (user_pos_r_embeds * item_r_embeds).sum(1)
        dot_product = dot_product - (user_neg_r_embeds * item_r_embeds).sum(1)

        prediction = dot_product + user_biases + item_biases

        if clip == 1:
            prediction = torch.clamp(prediction, 1, 5)

        return prediction

    def forward(self, **kwargs):
        user_id = kwargs["user_id"]
        item_id = kwargs["item_id"]
        clip = kwargs.get("clip", 0)
        return self._predict(user_id, item_id, clip=clip)

    def calculate_loss(self, **kwargs):
        user_id = kwargs["user_id"]
        item_id = kwargs["item_id"]
        rating = kwargs["rating"]

        prediction = self._predict(user_id, item_id, clip=0)
        return self.loss_fn(prediction, rating)
