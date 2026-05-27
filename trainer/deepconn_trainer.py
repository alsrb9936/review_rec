import torch
import torch.nn.functional as F

from trainer.base_trainer import BaseTrainer
from utils.metric import compute_all_metrics


class DeepCoNNTrainer(BaseTrainer):
    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad()
        rating = batch["rating"]
        user_reviews = batch["user_reviews"]
        item_reviews = batch["item_reviews"]


        loss = self.model.calculate_loss(user_reviews, item_reviews, rating)
        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, data_loader):
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)
                rating = batch["rating"]
                user_reviews = batch["user_reviews"]
                item_reviews = batch["item_reviews"]

                pred = self.model(user_reviews, item_reviews)

                all_preds.append(pred.cpu())
                all_targets.append(rating.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        return compute_all_metrics(all_preds, all_targets)

    def get_metric_name(self) -> str:
        return "rmse"
