import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel


class NARRE(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)



    def forward(self, batch):
        pass