import torch
import torch.nn.functional as F

from trainer.base_trainer import BaseTrainer
from utils.metric import compute_all_metrics


class NARRETrainer(BaseTrainer):
    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad()
        user_id = batch["user_id"]
        item_id = batch["item_id"]
        rating = batch["rating"]

        user_review = batch["user_reviews"]
        user_id_per_review = batch["user_review_item_ids"]
        item_review = batch["item_reviews"]
        item_id_per_review = batch["item_review_user_ids"]

        loss = self.model.calculate_loss(user_id, item_id, user_review, item_review, user_id_per_review, item_id_per_review, rating)
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
                user_id = batch["user_id"]
                item_id = batch["item_id"]
                rating = batch["rating"]

                user_reviews = batch["user_reviews"]
                item_reviews = batch["item_reviews"]
                user_id_per_review = batch["user_review_item_ids"]
                item_id_per_review = batch["item_review_user_ids"]

                pred = self.model(user_id, item_id, user_reviews, item_reviews, user_id_per_review, item_id_per_review)
                rating = rating.view(-1)
                all_preds.append(pred.cpu())
                all_targets.append(rating.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        return compute_all_metrics(all_preds, all_targets)

    def get_metric_name(self) -> str:
        return "rmse"
