import torch
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer
from utils.metric import compute_all_metrics


class NeuMFTrainer(BaseTrainer):
    def __init__(self, model, cfg, device):
        super().__init__(model, cfg, device)

    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad()

        user_id = batch["user_id"]
        item_id = batch["item_id"]
        rating = batch["rating"]

        loss = self.model.calculate_loss(user_id, item_id, rating)
        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, data_loader: DataLoader) -> dict:
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)
                user_id = batch["user_id"]
                item_id = batch["item_id"]
                rating = batch["rating"]

                preds = self.model(user_id, item_id)
                all_preds.append(preds.cpu())
                all_targets.append(rating.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        return compute_all_metrics(all_preds, all_targets)

    def get_metric_name(self) -> str:
        return "rmse"
