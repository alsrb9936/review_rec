import torch
from torch import nn
from torch.utils.data import DataLoader

from trainer.base_trainer import BaseTrainer
from utils.metric import compute_all_metrics


class LETTERTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, cfg, device: torch.device):
        super().__init__(model, cfg, device)

    def train_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.model.calculate_loss(batch["user_id"], batch["item_id"], batch["rating"])
        loss.backward()

        grad_clip = float(self.cfg.training.get("grad_clip", 1.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()
        return loss

    def evaluate(self, data_loader: DataLoader[dict[str, torch.Tensor]]) -> dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)
                preds = self.model(batch["user_id"], batch["item_id"], batch["rating"], clip=True)
                all_preds.append(preds.cpu())
                all_targets.append(batch["rating"].cpu())

        return compute_all_metrics(torch.cat(all_preds), torch.cat(all_targets))

    def get_metric_name(self) -> str:
        return "rmse"
