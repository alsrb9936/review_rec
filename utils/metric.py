import torch
import numpy as np


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean((predictions - targets) ** 2).item()


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean(torch.abs(predictions - targets)).item()


def compute_all_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    return {
        "rmse": rmse(predictions, targets),
        "mse": mse(predictions, targets),
        "mae": mae(predictions, targets),
    }
