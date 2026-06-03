import os

import torch

from trainer.base_trainer import BaseTrainer
from utils.metric import compute_all_metrics


class MultiOptimizerScheduler:
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()


class TransNetTrainer(BaseTrainer):
    def __init__(self, model, cfg, device):
        super().__init__(model, cfg, device)
        self.optimizers = {
            "act": self._build_optimizer(self.model.act_parameters()),
            "oth": self._build_optimizer(self.model.oth_parameters()),
            "full": self._build_optimizer(self.model.full_parameters()),
        }
        self.lr_scheduler = MultiOptimizerScheduler(
            [
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.lr_decay)
                for optimizer in self.optimizers.values()
            ]
        )

    def _build_optimizer(self, parameters):
        if self.cfg.training.optimizer == "Adam":
            return torch.optim.Adam(
                parameters,
                lr=self.cfg.training.lr,
                weight_decay=self.cfg.training.weight_decay,
            )
        return torch.optim.SGD(
            parameters,
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )

    def train_step(self, batch) -> torch.Tensor:
        self.model.train()

        if "target_doc" not in batch:
            raise KeyError("TransNet training requires target_doc in the batch.")

        losses = self.model.train_step(
            user_id=batch["user_id"],
            item_id=batch["item_id"],
            user_doc=batch["user_doc"],
            item_doc=batch["item_doc"],
            target_doc=batch["target_doc"],
            rating=batch["rating"],
            optimizers=self.optimizers,
        )
        return losses["loss"]

    def evaluate(self, data_loader):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)
                pred = self.model(
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

    def _save_checkpoint(self):
        os.makedirs(self.cfg.experiment.save_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.cfg.experiment.save_dir, f"{self.cfg.model_name}_best.pt"
        )
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dicts": {
                    name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
                },
                "best_metric": self.best_metric_value,
            },
            checkpoint_path,
        )

    def _load_checkpoint(self):
        checkpoint_path = os.path.join(
            self.cfg.experiment.save_dir, f"{self.cfg.model_name}_best.pt"
        )
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer_states = checkpoint.get("optimizer_state_dicts", {})
            for name, optimizer in self.optimizers.items():
                if name in optimizer_states:
                    optimizer.load_state_dict(optimizer_states[name])
            self.best_metric_value = checkpoint["best_metric"]
            self.current_epoch = checkpoint["epoch"]
