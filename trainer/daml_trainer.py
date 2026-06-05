import torch

from trainer.base_trainer import BaseTrainer
from utils.metric import compute_all_metrics


class DAMLTrainer(BaseTrainer):
    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.model.calculate_loss(
            user_id=batch["user_id"],
            item_id=batch["item_id"],
            user_doc=batch["user_doc"],
            item_doc=batch["item_doc"],
            rating=batch["rating"],
        )
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
                pred = self.model.predict(
                    batch["user_id"],
                    batch["item_id"],
                    batch["user_doc"],
                    batch["item_doc"],
                )
                all_preds.append(pred.view(-1).cpu())
                all_targets.append(batch["rating"].view(-1).cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        return compute_all_metrics(all_preds, all_targets)

    def get_metric_name(self) -> str:
        return "rmse"
