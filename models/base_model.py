import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

class BaseModel(nn.Module, abc.ABC):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def forward(self, **kwargs):
        ...

    @abc.abstractmethod
    def calculate_loss(self, **kwargs):
        ...

class TextCNN(nn.Module):
    def __init__(self, hyper_params):
        super(TextCNN, self).__init__()
        self.hyper_params = hyper_params

        self.num_filters = int(hyper_params["num_filters"])
        self.kernel_size = int(hyper_params["kernel_size"])
        self.word_embed_size = int(hyper_params["word_embed_size"])
        self.latent_size = int(hyper_params["latent_size"])

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.num_filters,
            kernel_size=(self.kernel_size, self.word_embed_size),
            padding=(self.kernel_size - 1, 0),
        )

        self.fc = nn.Linear(self.num_filters, self.latent_size)
        self.dropout = nn.Dropout(float(hyper_params["dropout"]))

    def forward(self, x):
        # x: [batch_size, num_reviews * num_words, word_embedding]
        x = torch.unsqueeze(x, 1)      # [B, 1, T, E]
        x = F.relu(self.conv(x))       # [B, num_filters, T', 1]
        x = torch.squeeze(x, -1)       # [B, num_filters, T']
        x = F.max_pool1d(x, x.size(2)) # [B, num_filters, 1]
        x = torch.squeeze(x, -1)       # [B, num_filters]
        x = self.dropout(self.fc(x))   # [B, latent_size]
        return x

class TorchFM(nn.Module):
    def __init__(self, n=None, k=None):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(n, k),requires_grad=True)
        self.lin = nn.Linear(n, 1)
        
    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        
        return out