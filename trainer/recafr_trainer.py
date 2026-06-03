import json
import logging
import math
import os

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm


class RecAFRTrainer:
    """Mini-batch BPR trainer with full-ranking Recall/NDCG evaluation."""

    def __init__(self, model, cfg, device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.best_metric_value = -float("inf")
        self.best_epoch = 0
        self.optimizer = self._build_optimizer()
        self.logger = self._build_logger()

    def _build_optimizer(self):
        optimizer_name = str(self.cfg.training.optimizer).lower()
        if optimizer_name == "adam":
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

    def train(self, train_loader, valid_loader, test_loader):
        patience_counter = 0
        best_test_metrics = None

        for epoch in range(1, int(self.cfg.training.epoch) + 1):
            self.model.train()
            total_loss = 0.0
            total_bpr = 0.0
            total_reg = 0.0
            total_kd = 0.0
            num_batches = 0

            pbar = tqdm(
                train_loader,
                desc=f"RecAFR [{epoch}/{int(self.cfg.training.epoch)}]",
                leave=False,
                dynamic_ncols=True,
            )
            for batch in pbar:
                batch = self._move_batch_to_device(batch)
                self.optimizer.zero_grad()
                loss, loss_dict = self.model.calculate_loss(
                    user_id=batch["user_id"],
                    pos_item_id=batch["pos_item_id"],
                    neg_item_id=batch["neg_item_id"],
                )
                loss.backward()

                grad_clip = float(self.cfg.training.get("grad_clip", 0.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()

                total_loss += float(loss_dict["loss"])
                total_bpr += float(loss_dict["bpr_loss"])
                total_reg += float(loss_dict["reg_loss"])
                total_kd += float(loss_dict["kd_loss"])
                num_batches += 1
                pbar.set_postfix(loss=f"{loss_dict['loss']:.4f}")

            if epoch % int(self.cfg.evaluation.eval_step) != 0:
                continue

            valid_metrics = self.evaluate(valid_loader)
            metric_name = self.get_metric_name()
            current_metric = float(valid_metrics.get(metric_name, -float("inf")))
            avg_loss = total_loss / max(num_batches, 1)
            avg_bpr = total_bpr / max(num_batches, 1)
            avg_reg = total_reg / max(num_batches, 1)
            avg_kd = total_kd / max(num_batches, 1)

            log_msg = (
                f"Epoch={epoch:>4d}, Loss={avg_loss:.4f}, BPR={avg_bpr:.4f}, "
                f"Reg={avg_reg:.6f}, KD={avg_kd:.4f}, "
                f"Valid {metric_name}={current_metric:.4f}"
            )

            if current_metric > self.best_metric_value:
                self.best_metric_value = current_metric
                self.best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint()
                best_test_metrics = self.evaluate(test_loader)
                log_msg += f", Test {metric_name}={best_test_metrics.get(metric_name, 0.0):.4f}"
            else:
                patience_counter += 1
                if patience_counter >= int(self.cfg.evaluation.early_stop_patience):
                    self.logger.info(log_msg)
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            self.logger.info(log_msg)

        self._load_checkpoint()
        test_metrics = best_test_metrics if best_test_metrics is not None else self.evaluate(test_loader)
        self.logger.info(
            f"Training complete. Best epoch={self.best_epoch}, "
            f"Best Valid {self.get_metric_name()}={self.best_metric_value:.4f}"
        )
        self.logger.info(f"Test Metrics: {test_metrics}")
        self._save_results(test_metrics)

    def evaluate(self, data_loader):
        dataset = data_loader.dataset
        k_values = list(self.cfg.evaluation.get("k", self.cfg.model.get("eval_k", [5, 10, 20])))
        k_values = [int(k) for k in k_values]
        max_k = max(k_values)

        recall_sum = {k: 0.0 for k in k_values}
        ndcg_sum = {k: 0.0 for k in k_values}
        eval_user_count = 0

        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)
                user_ids = batch["user_id"].view(-1)
                scores = self.model.predict_all(user_ids)

                for row_idx, user_id_tensor in enumerate(user_ids):
                    user_id = int(user_id_tensor.item())
                    train_items = dataset.train_user_pos.get(user_id, set())
                    target_items = dataset.eval_user_pos.get(user_id, set())
                    if len(target_items) == 0:
                        continue

                    if train_items:
                        mask_idx = torch.tensor(list(train_items), dtype=torch.long, device=scores.device)
                        scores[row_idx, mask_idx] = -1e9

                    top_items = torch.topk(scores[row_idx], k=max_k).indices.detach().cpu().tolist()
                    for k in k_values:
                        hits = [1 if item in target_items else 0 for item in top_items[:k]]
                        recall_sum[k] += sum(hits) / float(len(target_items))
                        ndcg_sum[k] += self._ndcg_at_k(hits, min(len(target_items), k))
                    eval_user_count += 1

        eval_user_count = max(eval_user_count, 1)
        metrics = {}
        for k in k_values:
            metrics[f"recall@{k}"] = recall_sum[k] / eval_user_count
            metrics[f"ndcg@{k}"] = ndcg_sum[k] / eval_user_count
        return metrics

    @staticmethod
    def _ndcg_at_k(hits, ideal_len: int) -> float:
        dcg = 0.0
        for idx, hit in enumerate(hits):
            if hit:
                dcg += 1.0 / math.log2(idx + 2.0)
        idcg = sum(1.0 / math.log2(idx + 2.0) for idx in range(ideal_len))
        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    def get_metric_name(self) -> str:
        k_values = list(self.cfg.evaluation.get("k", self.cfg.model.get("eval_k", [5, 10, 20])))
        return f"recall@{int(max(k_values))}"

    def _move_batch_to_device(self, batch):
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

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
                    "best_valid_metric_name": self.get_metric_name(),
                    "test_metrics": test_metrics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        self.logger.info(f"Test metrics saved to {result_path}")
