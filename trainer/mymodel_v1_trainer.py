# trainer/mymodel_trainer.py

import torch
from torch.utils.data import DataLoader

from trainer.base_trainer import BaseTrainer
from utils.metric import compute_all_metrics


class MyModelV1Trainer(BaseTrainer):
    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad()

        loss_dict = self.model.calculate_loss(
            user_id=batch["user_id"],
            item_id=batch["item_id"],
            rating=batch["rating"],
            return_dict=True,
        )

        loss = loss_dict["loss"]
        loss.backward()

        grad_clip = float(self.cfg.training.get("grad_clip", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        self.optimizer.step()
        return loss

    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)
                preds = self.model(
                    user_id=batch["user_id"],
                    item_id=batch["item_id"],
                )
                all_preds.append(preds.view(-1).cpu())
                all_targets.append(batch["rating"].view(-1).cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        return compute_all_metrics(all_preds, all_targets)

    def get_metric_name(self) -> str:
        return "rmse"