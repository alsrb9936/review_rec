import torch
from torch.utils.data import Dataset

class MyModelDataset(Dataset):
    def __init__(self, cfg, split: str = "train"):
        pass