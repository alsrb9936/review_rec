import json
import logging
import os

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from utils.metric import compute_all_metrics


class RecAFRTrainer:
    """Rating-prediction trainer for RecAFR."""

    def __init__(self, model, cfg, device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.best_metric_value = float("inf")
        self.best_epoch = 0
        self.optimizer = self._build_optimizer()
        self.logger = self._build_logger()

    def _build_optimizer(self):
        if str(self.cfg.training.optimizer).lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=float(self.cfg.training.lr),
                weight_decay=float(self.cfg.training.weight_decay),
            )
        return torch.optim.SGD(
            self.model.parameters(),
            lr=float(self.cfg.training.lr),
            weight_decay=float(self.cfg.training.weight_decay),
        )

    def _build_logger(self):
        os.makedirs(self.cfg.experiment.save_dir, exist_ok=True)
        OmegaConf.save(
            config=self.cfg,
            f=os.path.join(self.cfg.experiment.save_dir, "config.yaml"),
            resolve=True,
        )
        log_path = os.path.join(self.cfg.experiment.save_dir, "train.log")
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    def train(self, train_loader, valid_loader, test_loader):
        patience_counter = 0
        best_test_metrics = None

        for epoch in range(1, int(self.cfg.training.epoch) + 1):
            self.model.train()
            total_loss = 0.0
            total_mse = 0.0
            total_reg = 0.0
            total_kd = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"RecAFR [{epoch}/{int(self.cfg.training.epoch)}]", leave=False)
            for batch in pbar:
                batch = self._move_batch_to_device(batch)
                self.optimizer.zero_grad()
                loss, loss_dict = self.model.calculate_loss(
                    user_id=batch["user_id"],
                    item_id=batch["item_id"],
                    rating=batch["rating"],
                )
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss_dict["loss"])
                total_mse += float(loss_dict["mse_loss"])
                total_reg += float(loss_dict["reg_loss"])
                total_kd += float(loss_dict["kd_loss"])
                num_batches += 1
                pbar.set_postfix(loss=f"{loss_dict['loss']:.4f}")

            if epoch % int(self.cfg.evaluation.eval_step) != 0:
                continue

            valid_metrics = self.evaluate(valid_loader)
            valid_rmse = float(valid_metrics["rmse"])
            avg_loss = total_loss / max(num_batches, 1)
            avg_mse = total_mse / max(num_batches, 1)
            avg_reg = total_reg / max(num_batches, 1)
            avg_kd = total_kd / max(num_batches, 1)

            msg = (
                f"Epoch={epoch}, Loss={avg_loss:.4f}, MSELoss={avg_mse:.4f}, "
                f"Reg={avg_reg:.6f}, KD={avg_kd:.4f}, "
                f"Valid RMSE={valid_metrics['rmse']:.4f}, "
                f"MSE={valid_metrics['mse']:.4f}, MAE={valid_metrics['mae']:.4f}"
            )

            if valid_rmse < self.best_metric_value:
                self.best_metric_value = valid_rmse
                self.best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint()
                best_test_metrics = self.evaluate(test_loader)
                msg += f", Test RMSE={best_test_metrics['rmse']:.4f}"
            else:
                patience_counter += 1
                if patience_counter >= int(self.cfg.evaluation.early_stop_patience):
                    self.logger.info(msg)
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            self.logger.info(msg)

        self._load_checkpoint()
        test_metrics = best_test_metrics if best_test_metrics is not None else self.evaluate(test_loader)
        self.logger.info(f"Training complete. Best epoch={self.best_epoch}, Best Valid RMSE={self.best_metric_value:.4f}")
        self.logger.info(f"Test Metrics: {test_metrics}")
        self._save_results(test_metrics)

    def evaluate(self, data_loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)
                preds = self.model(user_id=batch["user_id"], item_id=batch["item_id"])
                all_preds.append(preds.view(-1).cpu())
                all_targets.append(batch["rating"].view(-1).cpu())
        return compute_all_metrics(torch.cat(all_preds), torch.cat(all_targets))

    def get_metric_name(self):
        return "rmse"

    def _move_batch_to_device(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _checkpoint_path(self):
        return os.path.join(self.cfg.experiment.save_dir, f"{self.cfg.model_name}_best.pt")

    def _save_checkpoint(self):
        torch.save(
            {
                "epoch": self.best_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_metric": self.best_metric_value,
            },
            self._checkpoint_path(),
        )

    def _load_checkpoint(self):
        path = self._checkpoint_path()
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_metric_value = checkpoint["best_metric"]
            self.best_epoch = checkpoint["epoch"]

    def _save_results(self, test_metrics):
        path = os.path.join(self.cfg.experiment.save_dir, "test_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_valid_metric": self.best_metric_value,
                    "best_valid_metric_name": "rmse",
                    "test_metrics": test_metrics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
