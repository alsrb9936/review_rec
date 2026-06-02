import torch
import numpy as np
from torch.nn import functional as F

def _flatten_pair(predictions: torch.Tensor, targets: torch.Tensor):
    predictions = predictions.view(-1).float()
    targets = targets.view(-1).float()

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: "
            f"predictions={predictions.shape}, targets={targets.shape}"
        )

    return predictions, targets


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    predictions, targets = _flatten_pair(predictions, targets)
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    predictions, targets = _flatten_pair(predictions, targets)
    return torch.mean((predictions - targets) ** 2).item()


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    predictions, targets = _flatten_pair(predictions, targets)
    return torch.mean(torch.abs(predictions - targets)).item()


def compute_all_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    predictions = predictions.view(-1).float()
    targets = targets.view(-1).float()

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: "
            f"predictions={predictions.shape}, targets={targets.shape}"
        )

    diff = predictions - targets

    return {
        "rmse": torch.sqrt(torch.mean(diff ** 2)).item(),
        "mse": torch.mean(diff ** 2).item(),
        "mae": torch.mean(torch.abs(diff)).item(),
    }