import csv
import json
import os
from typing import Any

import torch
from torch.utils.data import DataLoader

from trainer.base_trainer import BaseTrainer
from utils.metric import compute_all_metrics


class CFARGTrainer(BaseTrainer):
    DIAGNOSTIC_KEYS = [
        "user_cos",
        "item_cos",
        "user_cf_norm",
        "item_cf_norm",
        "user_review_proj_norm",
        "item_review_proj_norm",
        "user_injection_norm",
        "item_injection_norm",
        "user_injection_ratio",
        "item_injection_ratio",
        "effective_user_review_weight",
        "effective_item_review_weight",
    ]

    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad()
        loss_dict = self.model.calculate_loss(
            user_id=batch["user_id"],
            item_id=batch["item_id"],
            user_review=batch.get("user_review"),
            item_review=batch.get("item_review"),
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

    def evaluate(self, data_loader: DataLoader[dict[str, torch.Tensor]]) -> dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        user_gates = []
        item_gates = []
        review_scales = []
        diagnostics: dict[str, list[torch.Tensor]] = {key: [] for key in self.DIAGNOSTIC_KEYS}

        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)
                outputs = self.model(
                    user_id=batch["user_id"],
                    item_id=batch["item_id"],
                    user_review=batch.get("user_review"),
                    item_review=batch.get("item_review"),
                    return_dict=True,
                )
                all_preds.append(outputs["rating_pred"].view(-1).cpu())
                all_targets.append(batch["rating"].view(-1).cpu())
                user_gates.append(outputs["user_gate"].view(-1).cpu())
                item_gates.append(outputs["item_gate"].view(-1).cpu())

                if "review_scale" in outputs:
                    review_scales.append(outputs["review_scale"].view(-1).cpu())
                for key in self.DIAGNOSTIC_KEYS:
                    if key in outputs:
                        diagnostics[key].append(outputs[key].view(-1).detach().cpu())

        metrics = compute_all_metrics(torch.cat(all_preds), torch.cat(all_targets))
        metrics.update(self._gate_stats(torch.cat(user_gates), torch.cat(item_gates)))
        if review_scales:
            metrics["review_scale"] = float(torch.cat(review_scales).mean().item())

        for key, chunks in diagnostics.items():
            if chunks:
                metrics[f"avg_{key}"] = float(torch.cat(chunks).mean().item())
        return metrics

    def get_metric_name(self) -> str:
        return "rmse"

    def train(
        self,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        valid_loader: DataLoader[dict[str, torch.Tensor]],
        test_loader: DataLoader[dict[str, torch.Tensor]],
    ):
        for epoch in range(int(self.cfg.training.epoch)):
            self.current_epoch = epoch + 1
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                batch = self._move_batch_to_device(batch)
                loss = self.train_step(batch)
                total_loss += float(loss.detach().cpu().item())
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            self.lr_scheduler.step()

            if self.current_epoch % int(self.cfg.evaluation.eval_step) != 0:
                continue

            metrics = self.evaluate(valid_loader)
            current_metric = metrics.get(self.get_metric_name(), float("inf"))
            self.logger.info(
                f"Epoch [{self.current_epoch}/{self.cfg.training.epoch}] "
                f"Loss: {avg_loss:.4f} | Valid RMSE: {metrics.get('rmse', 0):.4f}, "
                f"MSE: {metrics.get('mse', 0):.4f}, MAE: {metrics.get('mae', 0):.4f}, "
                f"UserGate: {metrics.get('avg_user_gate', 0):.4f}, "
                f"ItemGate: {metrics.get('avg_item_gate', 0):.4f}, "
                f"ReviewScale: {metrics.get('review_scale', 0):.4f}, "
                f"UserCos: {metrics.get('avg_user_cos', 0):.4f}, "
                f"ItemCos: {metrics.get('avg_item_cos', 0):.4f}, "
                f"UserInjRatio: {metrics.get('avg_user_injection_ratio', 0):.4f}, "
                f"ItemInjRatio: {metrics.get('avg_item_injection_ratio', 0):.4f}"
            )

            if current_metric < self.best_metric_value:
                self.best_metric_value = current_metric
                self.patience_counter = 0
                self._save_checkpoint()
            else:
                self.patience_counter += 1

            if self.patience_counter >= int(self.cfg.evaluation.early_stop_patience):
                self.logger.info(f"Early stopping at epoch {self.current_epoch}")
                break

        self._load_checkpoint()
        test_metrics = self.evaluate(test_loader)
        self.logger.info(f"Test Metrics: {test_metrics}")
        self._save_test_results(test_metrics)
        self._save_experiment_results(test_metrics)
        self._save_gate_outputs(test_loader)

    @staticmethod
    def _gate_stats(user_gate: torch.Tensor, item_gate: torch.Tensor) -> dict[str, float]:
        return {
            "avg_user_gate": float(user_gate.mean().item()),
            "avg_item_gate": float(item_gate.mean().item()),
            "std_user_gate": float(user_gate.std(unbiased=False).item()),
            "std_item_gate": float(item_gate.std(unbiased=False).item()),
            "min_user_gate": float(user_gate.min().item()),
            "max_user_gate": float(user_gate.max().item()),
            "min_item_gate": float(item_gate.min().item()),
            "max_item_gate": float(item_gate.max().item()),
        }

    def _checkpoint_path(self) -> str:
        return os.path.join(self.cfg.experiment.save_dir, f"{self.cfg.model_name}_best.pt")

    def _save_test_results(self, test_metrics: dict[str, float]):
        path = os.path.join(self.cfg.experiment.save_dir, "test_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_valid_metric": self.best_metric_value,
                    "best_valid_metric_name": self.get_metric_name(),
                    "best_epoch": self.current_epoch,
                    "test_metrics": test_metrics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _base_result_row(self, test_metrics: dict[str, float]) -> dict[str, object]:
        config_path = os.path.join(self.cfg.experiment.save_dir, "config.yaml")
        noise_cfg = self.cfg.get("noise", {})
        row: dict[str, object] = {
            "dataset": str(self.cfg.data.dataset),
            "model": str(self.cfg.model.get("result_name", self.cfg.model.variant)),
            "seed": int(self.cfg.experiment.seed),
            "noise_type": str(noise_cfg.get("type", "none")) if bool(noise_cfg.get("enabled", False)) else "none",
            "noise_ratio": float(noise_cfg.get("ratio", 0.0)) if bool(noise_cfg.get("enabled", False)) else 0.0,
            "rmse": test_metrics.get("rmse"),
            "mae": test_metrics.get("mae"),
            "best_epoch": int(self.current_epoch),
            "config_path": config_path,
            "checkpoint_path": self._checkpoint_path(),
            "avg_user_gate": test_metrics.get("avg_user_gate"),
            "avg_item_gate": test_metrics.get("avg_item_gate"),
            "std_user_gate": test_metrics.get("std_user_gate"),
            "std_item_gate": test_metrics.get("std_item_gate"),
            "min_user_gate": test_metrics.get("min_user_gate"),
            "max_user_gate": test_metrics.get("max_user_gate"),
            "min_item_gate": test_metrics.get("min_item_gate"),
            "max_item_gate": test_metrics.get("max_item_gate"),
            "review_scale": test_metrics.get("review_scale"),
        }
        for key in self.DIAGNOSTIC_KEYS:
            row[f"avg_{key}"] = test_metrics.get(f"avg_{key}")
        return row

    def _save_experiment_results(self, test_metrics: dict[str, float]):
        results_dir = str(self.cfg.experiment.get("results_dir", "results"))
        os.makedirs(results_dir, exist_ok=True)
        result_file = str(self.cfg.experiment.get("result_file", "results/main_results.csv"))
        row = self._base_result_row(test_metrics)
        self._append_csv(result_file, row, self._columns_for(result_file))
        if str(self.cfg.model.get("variant", "")) == "gated" and os.path.basename(result_file) != "gate_stats.csv":
            gate_stats_path = os.path.join(results_dir, "gate_stats.csv")
            self._append_csv(gate_stats_path, row, self._columns_for(gate_stats_path))

    @classmethod
    def _columns_for(cls, path: str) -> list[str]:
        name = os.path.basename(path)
        main_columns = [
            "dataset", "model", "seed", "rmse", "mae", "best_epoch",
            "avg_user_gate", "avg_item_gate", "std_user_gate", "std_item_gate",
            "review_scale", "avg_user_cos", "avg_item_cos",
            "avg_user_injection_ratio", "avg_item_injection_ratio",
            "config_path", "checkpoint_path",
        ]
        if name == "main_results.csv":
            return main_columns
        if name == "noise_results.csv":
            return [
                "dataset", "model", "seed", "noise_type", "noise_ratio", "rmse", "mae",
                "avg_user_gate", "avg_item_gate", "std_user_gate", "std_item_gate",
                "review_scale", "avg_user_cos", "avg_item_cos",
                "avg_user_injection_ratio", "avg_item_injection_ratio",
            ]
        if name == "gate_stats.csv":
            return [
                "dataset", "model", "seed",
                "avg_user_gate", "avg_item_gate", "std_user_gate", "std_item_gate",
                "min_user_gate", "max_user_gate", "min_item_gate", "max_item_gate",
                "review_scale",
                "avg_user_cos", "avg_item_cos",
                "avg_user_cf_norm", "avg_item_cf_norm",
                "avg_user_review_proj_norm", "avg_item_review_proj_norm",
                "avg_user_injection_norm", "avg_item_injection_norm",
                "avg_user_injection_ratio", "avg_item_injection_ratio",
                "avg_effective_user_review_weight", "avg_effective_item_review_weight",
                "rmse", "mae", "checkpoint_path",
            ]
        return main_columns

    @staticmethod
    def _append_csv(path: str, row: dict[str, object], columns: list[str]):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerow({column: row.get(column, "") for column in columns})

    @staticmethod
    def _output_value(outputs: dict[str, Any], key: str, idx: int) -> object:
        value = outputs.get(key)
        if value is None:
            return ""
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.numel() == 1:
                return float(value.item())
            return float(value.view(-1)[idx].item())
        return value

    def _save_gate_outputs(self, data_loader: DataLoader[dict[str, torch.Tensor]]):
        if str(self.cfg.model.get("variant", "")) != "gated":
            return
        results_dir = str(self.cfg.experiment.get("results_dir", "results"))
        os.makedirs(results_dir, exist_ok=True)
        sample_limit = int(self.cfg.evaluation.get("gate_sample_size", 1000))
        rows = []
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_batch_to_device(batch)
                outputs = self.model(
                    batch["user_id"],
                    batch["item_id"],
                    batch.get("user_review"),
                    batch.get("item_review"),
                    return_dict=True,
                )
                for idx in range(len(outputs["rating_pred"])):
                    rows.append(
                        {
                            "dataset": str(self.cfg.data.dataset),
                            "model": str(self.cfg.model.get("result_name", self.cfg.model.variant)),
                            "seed": int(self.cfg.experiment.seed),
                            "user_id": int(batch["user_id"][idx].detach().cpu().item()),
                            "item_id": int(batch["item_id"][idx].detach().cpu().item()),
                            "rating": float(batch["rating"][idx].detach().cpu().item()),
                            "prediction": float(outputs["rating_pred"][idx].detach().cpu().item()),
                            "user_gate": self._output_value(outputs, "user_gate", idx),
                            "item_gate": self._output_value(outputs, "item_gate", idx),
                            "review_scale": self._output_value(outputs, "review_scale", idx),
                            "user_cos": self._output_value(outputs, "user_cos", idx),
                            "item_cos": self._output_value(outputs, "item_cos", idx),
                            "user_injection_ratio": self._output_value(outputs, "user_injection_ratio", idx),
                            "item_injection_ratio": self._output_value(outputs, "item_injection_ratio", idx),
                            "effective_user_review_weight": self._output_value(outputs, "effective_user_review_weight", idx),
                            "effective_item_review_weight": self._output_value(outputs, "effective_item_review_weight", idx),
                        }
                    )
                    if len(rows) >= sample_limit:
                        break
                if len(rows) >= sample_limit:
                    break

        path = os.path.join(results_dir, "gate_values_sample.csv")
        columns = [
            "dataset", "model", "seed", "user_id", "item_id", "rating", "prediction",
            "user_gate", "item_gate", "review_scale",
            "user_cos", "item_cos",
            "user_injection_ratio", "item_injection_ratio",
            "effective_user_review_weight", "effective_item_review_weight",
        ]
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
