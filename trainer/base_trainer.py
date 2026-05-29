import abc
import json
import logging
import os
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class BaseTrainer(abc.ABC):
    def __init__(self, model: nn.Module, cfg: DictConfig, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device

        os.makedirs(cfg.experiment.save_dir, exist_ok=True)
        log_path = os.path.join(cfg.experiment.save_dir, "train.log")
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        self.logger = logger
        self.logger.info(f"Save directory: {cfg.experiment.save_dir}")

        if cfg.training.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
            )

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=cfg.training.lr_decay
        )

        self.best_metric_value = float("inf")
        self.patience_counter = 0
        self.current_epoch = 0

    @abc.abstractmethod
    def train_step(self, batch) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def evaluate(self, data_loader: DataLoader) -> dict:
        ...

    @abc.abstractmethod
    def get_metric_name(self) -> str:
        ...

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader):
        for epoch in range(self.cfg.training.epoch):
            self.current_epoch = epoch + 1
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch [{self.current_epoch}/{self.cfg.training.epoch}]",
                leave=False,
                dynamic_ncols=True,
            )
            for batch in pbar:
                batch = self._move_batch_to_device(batch)
                loss = self.train_step(batch)
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / max(num_batches, 1)
            self.lr_scheduler.step()
        

            if self.current_epoch % self.cfg.evaluation.eval_step == 0:
                metrics = self.evaluate(valid_loader)
                metric_name = self.get_metric_name()
                current_metric = metrics.get(metric_name, float("inf"))
                
                self.logger.info(
                    f"Epoch [{self.current_epoch}/{self.cfg.training.epoch}] "
                    f"Loss: {avg_loss:.4f} | Valid RMSE: {metrics.get('rmse', 0):.4f}, "
                    f"MSE: {metrics.get('mse', 0):.4f}, MAE: {metrics.get('mae', 0):.4f}"
                )

                if current_metric < self.best_metric_value:
                    print(f"New best {metric_name}: {current_metric:.4f} (previous: {self.best_metric_value:.4f})")
                    self.best_metric_value = current_metric
                    self.patience_counter = 0
                    self._save_checkpoint()
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.cfg.evaluation.early_stop_patience:
                    self.logger.info(f"Early stopping at epoch {self.current_epoch}")
                    break

        self.logger.info(f"Training complete. Best {self.get_metric_name()}: {self.best_metric_value:.4f}")
        self._load_checkpoint()
        test_metrics = self.evaluate(test_loader)
        self.logger.info(f"Test Metrics: {test_metrics}")

        test_metrics_path = os.path.join(self.cfg.experiment.save_dir, "test_results.json")
        with open(test_metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_valid_metric": self.best_metric_value,
                    "best_valid_metric_name": self.get_metric_name(),
                    "test_metrics": test_metrics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        self.logger.info(f"Test metrics saved to {test_metrics_path}")

    def _move_batch_to_device(self, batch):
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _save_checkpoint(self):
        os.makedirs(self.cfg.experiment.save_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.cfg.experiment.save_dir, f"{self.cfg.model_name}_best.pt"
        )
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric_value,
        }, checkpoint_path)

    def _load_checkpoint(self):
        checkpoint_path = os.path.join(
            self.cfg.experiment.save_dir, f"{self.cfg.model_name}_best.pt"
        )
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_metric_value = checkpoint["best_metric"]
            self.current_epoch = checkpoint["epoch"]
