# trainer/rgcl_trainer.py
import torch
from torch.utils.data import DataLoader

from trainer.base_trainer import BaseTrainer
from utils.metric import compute_all_metrics


class RGCLTrainer(BaseTrainer):
    def __init__(self, model, cfg, device):
        super().__init__(model, cfg, device)
        self.model.set_graph_device(device)

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model.calculate_loss(
            user_id=batch["user_id"],
            item_id=batch["item_id"],
            rating=batch["rating"],
            review_feat=batch["review_feat"],
        )

        loss = outputs["loss"]
        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, data_loader: DataLoader) -> dict:
        torch.cuda.empty_cache()
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)

                # valid/test에서는 target review를 사용하지 않음
                preds = self.model(
                    user_id=batch["user_id"],
                    item_id=batch["item_id"],
                )

                all_preds.append(preds.cpu())
                all_targets.append(batch["rating"].cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        return compute_all_metrics(all_preds, all_targets)

    def get_metric_name(self):
        return "rmse"