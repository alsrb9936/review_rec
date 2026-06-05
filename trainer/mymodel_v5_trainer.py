# trainer/mymodel_trainer.py

import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from trainer.base_trainer import BaseTrainer
from utils.metric import compute_all_metrics


class MyModelV5Trainer(BaseTrainer):
    LOG_KEYS = [
        "loss",
        "rating_loss",
        "pair_align_loss",
        "weighted_pair_align_loss",
        "pos_sim",
        "neg_sim",
        "sim_gap",
        "orthogonal_residual_cos",
        "orthogonal_error",
        "shared_norm",
        "orthogonal_norm",
        "shared_ratio",
        "orthogonal_ratio",
        "valid_review_ratio",
        "shared_raw_cos",
    ]

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
    ):
        self.logger.info(
            f"MyModelV5Trainer started | "
            f"lambda_pair_align={getattr(self.model, 'lambda_pair_align', None)}, "
            f"orthogonal_residual_weight={getattr(self.model, 'orthogonal_residual_weight', None)}, "
            f"temperature={getattr(self.model, 'temperature', None)}, "
            f"subspace_rank={getattr(self.model, 'subspace_rank', None)}"
        )

        for epoch in range(int(self.cfg.training.epoch)):
            self.current_epoch = epoch + 1
            self.model.train()

            total_loss = 0.0
            num_batches = 0
            loss_sums = {}

            pbar = tqdm(
                train_loader,
                desc=f"Epoch [{self.current_epoch}/{self.cfg.training.epoch}]",
                leave=False,
                dynamic_ncols=True,
            )

            for batch in pbar:
                batch = self._move_batch_to_device(batch)

                loss, loss_log = self.train_step(batch)

                loss_value = float(loss.detach().cpu().item())
                total_loss += loss_value
                num_batches += 1

                for key, value in loss_log.items():
                    loss_sums[key] = loss_sums.get(key, 0.0) + float(value)

                pbar.set_postfix(
                    loss=f"{loss_log.get('loss', 0.0):.4f}",
                    rating=f"{loss_log.get('rating_loss', 0.0):.4f}",
                    pair=f"{loss_log.get('pair_align_loss', 0.0):.4f}",
                    wp=f"{loss_log.get('weighted_pair_align_loss', 0.0):.4f}",
                    pos=f"{loss_log.get('pos_sim', 0.0):.3f}",
                    neg=f"{loss_log.get('neg_sim', 0.0):.3f}",
                    gap=f"{loss_log.get('sim_gap', 0.0):.3f}",
                    shr=f"{loss_log.get('shared_ratio', 0.0):.3f}",
                    ort=f"{loss_log.get('orthogonal_ratio', 0.0):.3f}",
                )

            avg_loss = total_loss / max(num_batches, 1)
            avg_loss_log = {
                key: value / max(num_batches, 1)
                for key, value in loss_sums.items()
            }

            self.lr_scheduler.step()

            if self.current_epoch % int(self.cfg.evaluation.eval_step) != 0:
                continue

            metrics = self.evaluate(valid_loader)
            metric_name = self.get_metric_name()
            current_metric = metrics.get(metric_name, float("inf"))

            self._log_epoch(
                avg_loss=avg_loss,
                avg_loss_log=avg_loss_log,
                metrics=metrics,
            )

            if current_metric < self.best_metric_value:
                self.logger.info(
                    f"New best {metric_name}: "
                    f"{current_metric:.4f} "
                    f"(previous: {self.best_metric_value:.4f})"
                )
                self.best_metric_value = current_metric
                self.patience_counter = 0
                self._save_checkpoint()
            else:
                self.patience_counter += 1

            if self.patience_counter >= int(self.cfg.evaluation.early_stop_patience):
                self.logger.info(f"Early stopping at epoch {self.current_epoch}")
                break

        self.logger.info(
            f"Training complete. "
            f"Best {self.get_metric_name()}: {self.best_metric_value:.4f}"
        )

        self._load_checkpoint()

        test_metrics = self.evaluate(test_loader)
        self.logger.info(f"Test Metrics: {test_metrics}")

        self._save_test_results(test_metrics)

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        loss_dict = self.model.calculate_loss(
            user_id=batch["user_id"],
            item_id=batch["item_id"],
            rating=batch["rating"],
            review_emb=batch.get("review_emb", None),
            return_dict=True,
        )

        if "loss" not in loss_dict:
            raise KeyError(
                "model.calculate_loss(..., return_dict=True) must return a dict "
                "containing key 'loss'."
            )

        loss = loss_dict["loss"]
        loss.backward()

        grad_clip = float(self.cfg.training.get("grad_clip", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        self.optimizer.step()

        loss_log = self._tensor_dict_to_float(loss_dict)

        return loss, loss_log

    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)

                # Review-free inference.
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

    def _tensor_dict_to_float(self, loss_dict: dict) -> dict[str, float]:
        log_dict = {}

        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                detached = value.detach()
                if detached.numel() == 1:
                    log_dict[key] = float(detached.cpu().item())
                else:
                    log_dict[key] = float(detached.mean().cpu().item())
            else:
                log_dict[key] = float(value)

        return log_dict

    def _log_epoch(
        self,
        avg_loss: float,
        avg_loss_log: dict[str, float],
        metrics: dict[str, float],
    ):
        self.logger.info(
            f"Epoch [{self.current_epoch}/{self.cfg.training.epoch}] "
            f"Loss: {avg_loss_log.get('loss', avg_loss):.4f}, "
            f"Rating: {avg_loss_log.get('rating_loss', 0.0):.4f}, "
            f"PairAlign: {avg_loss_log.get('pair_align_loss', 0.0):.4f}, "
            f"WeightedPair: {avg_loss_log.get('weighted_pair_align_loss', 0.0):.4f}, "
            f"PosSim: {avg_loss_log.get('pos_sim', 0.0):.4f}, "
            f"NegSim: {avg_loss_log.get('neg_sim', 0.0):.4f}, "
            f"SimGap: {avg_loss_log.get('sim_gap', 0.0):.4f}, "
            f"SharedRatio: {avg_loss_log.get('shared_ratio', 0.0):.4f}, "
            f"OrthRatio: {avg_loss_log.get('orthogonal_ratio', 0.0):.4f}, "
            f"SharedRawCos: {avg_loss_log.get('shared_raw_cos', 0.0):.4f}, "
            f"SharedNorm: {avg_loss_log.get('shared_norm', 0.0):.4f}, "
            f"OrthNorm: {avg_loss_log.get('orthogonal_norm', 0.0):.4f}, "
            f"OrthErr: {avg_loss_log.get('orthogonal_error', 0.0):.6f}, "
            f"ValidReview: {avg_loss_log.get('valid_review_ratio', 0.0):.4f} | "
            f"Valid RMSE: {metrics.get('rmse', 0):.4f}, "
            f"MSE: {metrics.get('mse', 0):.4f}, "
            f"MAE: {metrics.get('mae', 0):.4f}"
            f"ReviewRating: {avg_loss_log.get('review_rating_loss', 0.0):.4f}, "
            f"WReviewRating: {avg_loss_log.get('weighted_review_rating_loss', 0.0):.4f}, "
            f"ReviewRMSE: {avg_loss_log.get('review_rmse', 0.0):.4f}, "
            f"Distill: {avg_loss_log.get('distill_loss', 0.0):.4f}, "
            f"WDistill: {avg_loss_log.get('weighted_distill_loss', 0.0):.4f}, "
        )

    def _save_test_results(self, test_metrics: dict[str, float]):
        test_metrics_path = os.path.join(
            self.cfg.experiment.save_dir,
            "test_results.json",
        )

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

