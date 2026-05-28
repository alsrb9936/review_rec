import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

class CNN(nn.Module):

    def __init__(self, cfg, word_dim):
        super(CNN, self).__init__()

        self.num_filters = cfg.model.num_filters
        self.review_count = cfg.data.review_count
        self.review_length = cfg.data.review_length

        self.hidden_dim = cfg.model.hidden_dim
        self.kernel_size = cfg.model.kernel_size
        self.dropout = cfg.model.dropout

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=word_dim,
                out_channels=self.num_filters,
                kernel_size=self.kernel_size,
                padding=(self.kernel_size - 1) // 2),  # out shape(new_batch_size, model.num_filters, review_length)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, self.review_length)),  # out shape(new_batch_size,model.num_filters,1)
            nn.Dropout(p=self.dropout))

        self.linear = nn.Sequential(
            nn.Linear(self.num_filters * self.review_count, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout))

    def forward(self, vec):  # input shape(new_batch_size, review_length, word2vec_dim)
        latent = self.conv(vec.permute(0, 2, 1))  # out(new_batch_size, model.num_filters, 1) kernel count指一条评论潜在向量
        latent = self.linear(latent.reshape(-1, self.num_filters * self.review_count))
        return latent  # out shape(batch_size, hidden_dim)


class FactorizationMachine(nn.Module):

    def __init__(self, p, k):  # p=hidden_dim
        super().__init__()
        self.v = nn.Parameter(torch.rand(p, k) / 10)
        self.linear = nn.Linear(p, 1, bias=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        linear_part = self.linear(x)  # input shape(batch_size, hidden_dim), out shape(batch_size, 1)
        inter_part1 = torch.mm(x, self.v) ** 2
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True)
        pair_interactions = self.dropout(pair_interactions)
        output = linear_part + 0.5 * pair_interactions
        return output  # out shape(batch_size, 1)



class DeepCoNN(nn.Module):
    def __init__(self, cfg, word_emb):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.cnn_u = CNN(cfg, word_dim=self.embedding.embedding_dim)
        self.cnn_i = CNN(cfg, word_dim=self.embedding.embedding_dim)
        self.fm = FactorizationMachine(cfg.model.hidden_dim * 2, 10)
        self.lossfn = nn.MSELoss(reduction='sum')

    def forward(self, user_review, item_review):  # input shape(batch_size, review_count, review_length)
        new_batch_size = user_review.shape[0] * user_review.shape[1]
        user_review = user_review.reshape(new_batch_size, -1)
        item_review = item_review.reshape(new_batch_size, -1)

        u_vec = self.embedding(user_review)
        i_vec = self.embedding(item_review)

        user_latent = self.cnn_u(u_vec)
        item_latent = self.cnn_i(i_vec)

        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        prediction = self.fm(concat_latent)
        return prediction

    def calculate_loss(self, user_reviews, item_reviews, rating):
        prediction = self.forward(user_reviews, item_reviews)
        return self.lossfn(prediction, rating.view(-1, 1).float())