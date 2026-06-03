import json
import logging
import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from utils.metric import compute_all_metrics


class RGCLTrainer:
    """Full-batch trainer for the original-style RGCL implementation."""

    def __init__(self, model, cfg, device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.optimizer = None
        self.best_metric_value = float("inf")
        self.best_epoch = 0
        self.logger = self._build_logger()

    def _build_logger(self):
        os.makedirs(self.cfg.experiment.save_dir, exist_ok=True)
        config_path = os.path.join(self.cfg.experiment.save_dir, "config.yaml")
        OmegaConf.save(config=self.cfg, f=config_path, resolve=True)

        log_path = os.path.join(self.cfg.experiment.save_dir, "train.log")
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.info(f"Save directory: {self.cfg.experiment.save_dir}")
        return logger

    def _configure(self, dataset):
        dataset.to(self.device)
        self.model.configure_from_dataset(dataset)
        self.model.to(self.device)

        if self.cfg.training.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(self.cfg.training.lr),
                weight_decay=float(self.cfg.training.weight_decay),
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=float(self.cfg.training.lr),
                weight_decay=float(self.cfg.training.weight_decay),
            )

    def train(self, train_dataset, valid_dataset=None, test_dataset=None):
        dataset = train_dataset
        self._configure(dataset)

        if self.model.train_classification:
            rating_loss_net = nn.CrossEntropyLoss()
            train_gt_labels = dataset.train_labels
        else:
            rating_loss_net = nn.MSELoss()
            train_gt_labels = dataset.train_truths.float()
        train_gt_ratings = dataset.train_truths.float()

        learning_rate = float(self.cfg.training.lr)
        no_better_valid = 0
        best_test_metrics = None

        pbar = tqdm(
            range(1, int(self.cfg.training.epoch) + 1),
            desc=f"RGCL [{int(self.cfg.training.epoch)} iters]",
            leave=False,
            dynamic_ncols=True,
        )
        for iter_idx in pbar:
            self.model.train()
            if self.optimizer is None:
                raise RuntimeError("RGCL optimizer was not initialized.")
            optimizer = self.optimizer
            optimizer.zero_grad()

            pred_ratings1, ed_mi1, user1, item1 = self.model(
                dataset.train_enc_graph,
                dataset.train_dec_graph,
                dataset.user_feature,
                dataset.movie_feature,
                cal_edge_mi=True,
            )
            pred_ratings2, ed_mi2, user2, item2 = self.model(
                dataset.train_enc_graph,
                dataset.train_dec_graph,
                dataset.user_feature,
                dataset.movie_feature,
                cal_edge_mi=True,
            )

            loss1 = rating_loss_net(pred_ratings1, train_gt_labels).mean()
            loss2 = rating_loss_net(pred_ratings2, train_gt_labels).mean()
            rating_loss = (loss1 + loss2) / 2.0

            user_mi_loss = self.model.contrast_loss(user1, user2).mean()
            item_mi_loss = self.model.contrast_loss(item1, item2).mean()
            nd_loss = (user_mi_loss + item_mi_loss) / 2.0
            ed_loss = (ed_mi1.mean() + ed_mi2.mean()) / 2.0

            total_loss = rating_loss + self.model.lambda_nd * nd_loss + self.model.lambda_ed * ed_loss
            total_loss.backward()

            grad_clip = float(self.cfg.training.get("grad_clip", 1.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()

            train_pred = self.model.expected_rating(pred_ratings1)
            train_rmse = torch.sqrt(((train_pred - train_gt_ratings) ** 2).mean())

            if iter_idx % int(self.cfg.evaluation.eval_step) != 0:
                pbar.set_postfix(loss=f"{total_loss.item():.4f}")
                continue

            valid_metrics = self.evaluate(dataset, segment="valid")
            valid_rmse = float(valid_metrics["rmse"])
            log_msg = (
                f"Iter={iter_idx:>4d}, Loss={total_loss.item():.4f}, "
                f"Train_RMSE={train_rmse.item():.4f}, ED_MI={ed_loss.item():.4f}, "
                f"ND_MI={nd_loss.item():.4f}, Valid_RMSE={valid_rmse:.4f}"
            )

            if valid_rmse < self.best_metric_value:
                self.best_metric_value = valid_rmse
                self.best_epoch = iter_idx
                no_better_valid = 0
                self._save_checkpoint()
                best_test_metrics = self.evaluate(dataset, segment="test")
                log_msg += f", Test_RMSE={best_test_metrics['rmse']:.4f}"
            else:
                no_better_valid += 1
                if no_better_valid > int(self.cfg.training.get("lr_decay_patience", 20)):
                    new_lr = max(
                        learning_rate * float(self.cfg.training.get("lr_decay_factor", self.cfg.training.lr_decay)),
                        float(self.cfg.training.get("min_lr", 0.001)),
                    )
                    if new_lr < learning_rate:
                        learning_rate = new_lr
                        for group in optimizer.param_groups:
                            group["lr"] = learning_rate
                        no_better_valid = 0
                        log_msg += f", LR={learning_rate:g}"

                if no_better_valid >= int(self.cfg.evaluation.early_stop_patience):
                    self.logger.info(log_msg)
                    self.logger.info(f"Early stopping at iter {iter_idx}")
                    break

            self.logger.info(log_msg)
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", valid_rmse=f"{valid_rmse:.4f}")

        self._load_checkpoint()
        test_metrics = self.evaluate(dataset, segment="test")
        if best_test_metrics is not None:
            test_metrics = best_test_metrics

        self.logger.info(
            f"Training complete. Best iter={self.best_epoch}, Best Valid RMSE={self.best_metric_value:.4f}"
        )
        self.logger.info(f"Test Metrics: {test_metrics}")
        self._save_results(test_metrics)

    def evaluate(self, dataset, segment="valid"):
        if segment == "valid":
            enc_graph = dataset.valid_enc_graph
            dec_graph = dataset.valid_dec_graph
            rating_values = dataset.valid_truths
        elif segment == "test":
            enc_graph = dataset.test_enc_graph
            dec_graph = dataset.test_dec_graph
            rating_values = dataset.test_truths
        else:
            raise ValueError(f"Unsupported segment: {segment}")

        self.model.eval()
        with torch.no_grad():
            pred_ratings, _, _ = self.model(
                enc_graph,
                dec_graph,
                dataset.user_feature,
                dataset.movie_feature,
                cal_edge_mi=False,
            )
            predictions = self.model.expected_rating(pred_ratings)
        return compute_all_metrics(predictions.detach().cpu(), rating_values.detach().cpu())

    def _checkpoint_path(self):
        return os.path.join(self.cfg.experiment.save_dir, f"{self.cfg.model_name}_best.pt")

    def _save_checkpoint(self):
        if self.optimizer is None:
            raise RuntimeError("RGCL optimizer was not initialized.")
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
        if self.optimizer is None:
            raise RuntimeError("RGCL optimizer was not initialized.")
        checkpoint_path = self._checkpoint_path()
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_metric_value = checkpoint["best_metric"]
            self.best_epoch = checkpoint["epoch"]

    def _save_results(self, test_metrics):
        result_path = os.path.join(self.cfg.experiment.save_dir, "test_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
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
        self.logger.info(f"Test metrics saved to {result_path}")
