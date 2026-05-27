import abc
import torch
import torch.nn as nn
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
