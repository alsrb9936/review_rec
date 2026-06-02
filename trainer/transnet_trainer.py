import torch

from trainer.base_trainer import BaseTrainer
from utils.metric import compute_all_metrics


class TransNetTrainer(BaseTrainer):
    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad()

        if "target_doc" not in batch:
            raise KeyError("TransNet training requires target_doc in the batch.")

        loss = self.model.calculate_loss(
            user_doc=batch["user_doc"],
            item_doc=batch["item_doc"],
            target_doc=batch["target_doc"],
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
                pred = self.model(
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
