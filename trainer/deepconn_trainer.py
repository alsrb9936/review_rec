import torch
import torch.nn.functional as F

from trainer.base_trainer import BaseTrainer
from utils.metric import calculate_rating_metrics


class DeepCoNNTrainer(BaseTrainer):
    def train_step(self, batch) -> torch.Tensor:
        self.optimizer.zero_grad()

        pred = self.model(batch)
        rating = batch["rating"]

        loss = F.mse_loss(pred, rating)
        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, data_loader):
        self.model.eval()

        preds = []
        labels = []

        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)
                pred = self.model(batch)

                preds.append(pred.detach().cpu())
                labels.append(batch["rating"].detach().cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        mse = F.mse_loss(preds, labels).item()
        mae = F.l1_loss(preds, labels).item()
        rmse = mse ** 0.5

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
        }

    def get_metric_name(self) -> str:
        return "rmse"