import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, TensorDataset

from models.mymodel_v4 import MyModelV4
from utils.metric import compute_all_metrics
from utils.utils import set_stats_from_npy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze mymodel_v4 sentiment examples with prediction error, "
            "review residual ratio, and original review text."
)
    )
    parser.add_argument("--checkpoint", required=True, help="Path to mymodel_v4_best.pt")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. Amazon_Digital_Music_14")
    parser.add_argument("--subset", default="sentiment_pos", choices=["sentiment_pos", "sentiment_neg"])
    parser.add_argument("--split", default="test", choices=["test"], help="Only test is supported because sentiment ids align to test rows")
    parser.add_argument("--data-root", default="./dataset")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--device", default="0", help="GPU id or cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20, help="Number of examples per group/sort")
    parser.add_argument("--config", default=None, help="Optional config.yaml. Defaults to checkpoint directory config.yaml")
    parser.add_argument(
        "--review-source",
        default="auto",
        choices=["auto", "model_fallback", "user_item_average", "test_review_emb"],
        help=(
            "Review embedding source for residual analysis. "
            "auto uses test_review_emb.npy if present, otherwise user/item train-review profiles; "
            "model_fallback matches existing mymodel_v4 eval behavior but often gives zero residuals on test pairs; "
            "user_item_average uses saved user/item train-review profiles; "
            "test_review_emb requires test_review_emb.npy."
        ),
    )
    return parser.parse_args()


def load_cfg(args: argparse.Namespace):
    checkpoint_dir = Path(args.checkpoint).resolve().parent
    candidate_config = Path(args.config) if args.config else checkpoint_dir / "config.yaml"

    if candidate_config.exists():
        cfg = OmegaConf.load(candidate_config)
    else:
        cfg = OmegaConf.merge(
            OmegaConf.load("configs/config.yaml"),
            OmegaConf.load("configs/model/mymodel_v4.yaml"),
        )

    with open_dict(cfg):
        cfg.model_name = "mymodel_v4"
        cfg.data.dataset = args.dataset
        cfg.data.root = args.data_root
        cfg.data.type = "bert"
        cfg.experiment.device = 0 if args.device == "cpu" else int(args.device)

    return set_stats_from_npy(cfg)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{int(device_arg)}")


def load_test_arrays(
    data_dir: Path,
    review_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], str]:
    user_ids = np.load(data_dir / "test_user_id.npy").astype(np.int64)
    item_ids = np.load(data_dir / "test_item_id.npy").astype(np.int64)
    ratings = np.load(data_dir / "test_rating.npy").astype(np.float32)

    review_emb = None
    resolved_source = review_source
    if review_source == "auto":
        test_review_path = data_dir / "test_review_emb.npy"
        user_review_path = data_dir / "user_review_emb.npy"
        item_review_path = data_dir / "item_review_emb.npy"
        if test_review_path.exists():
            review_emb = np.load(test_review_path).astype(np.float32)
            resolved_source = "test_review_emb"
        elif user_review_path.exists() and item_review_path.exists():
            user_review_emb = np.load(user_review_path).astype(np.float32)
            item_review_emb = np.load(item_review_path).astype(np.float32)
            review_emb = 0.5 * (user_review_emb[user_ids] + item_review_emb[item_ids])
            resolved_source = "user_item_average"
        else:
            review_emb = None
            resolved_source = "model_fallback"
    elif review_source == "model_fallback":
        review_emb = None
    elif review_source == "test_review_emb":
        review_path = data_dir / "test_review_emb.npy"
        if not review_path.exists():
            raise FileNotFoundError(
                f"Missing {review_path}. Current BERT preprocessing usually does not save test review embeddings. "
                "Use --review-source model_fallback or --review-source user_item_average."
            )
        review_emb = np.load(review_path).astype(np.float32)
    elif review_source == "user_item_average":
        user_review_path = data_dir / "user_review_emb.npy"
        item_review_path = data_dir / "item_review_emb.npy"
        if not user_review_path.exists() or not item_review_path.exists():
            raise FileNotFoundError(
                f"Missing {user_review_path} or {item_review_path} for user_item_average review source."
            )
        user_review_emb = np.load(user_review_path).astype(np.float32)
        item_review_emb = np.load(item_review_path).astype(np.float32)
        review_emb = 0.5 * (user_review_emb[user_ids] + item_review_emb[item_ids])
    else:
        raise ValueError(f"Unknown review source: {review_source}")

    if review_emb is not None and not (len(user_ids) == len(item_ids) == len(ratings) == len(review_emb)):
        raise ValueError(
            "test arrays must align: "
            f"users={len(user_ids)}, items={len(item_ids)}, ratings={len(ratings)}, reviews={len(review_emb)}"
        )
    if review_emb is None and not (len(user_ids) == len(item_ids) == len(ratings)):
        raise ValueError(
            "test arrays must align: "
            f"users={len(user_ids)}, items={len(item_ids)}, ratings={len(ratings)}"
        )
    return user_ids, item_ids, ratings, review_emb, resolved_source


def load_subset_ids(dataset_root: Path, subset: str) -> dict[str, np.ndarray]:
    subset_dir = dataset_root / subset
    return {
        "consistent": np.load(subset_dir / "consistent_id.npy").astype(np.int64),
        "inconsistent": np.load(subset_dir / "inconsistent_id.npy").astype(np.int64),
    }


def load_text_frame(dataset_root: Path, subset: str) -> pd.DataFrame:
    sentiment_csv = dataset_root / "sentiment_test_predictions.csv"
    common_test_csv = dataset_root / "common" / "test.csv"

    if sentiment_csv.exists():
        df = pd.read_csv(sentiment_csv)
    elif common_test_csv.exists():
        df = pd.read_csv(common_test_csv)
        df.insert(0, "test_id", np.arange(len(df), dtype=np.int64))
    else:
        raise FileNotFoundError(
            f"Missing review text source. Expected {sentiment_csv} or {common_test_csv}."
        )

    if "test_id" not in df.columns:
        df.insert(0, "test_id", np.arange(len(df), dtype=np.int64))
    if f"{subset}_status" not in df.columns:
        df[f"{subset}_status"] = "unknown"
    return df


def load_model(cfg, checkpoint_path: str, device: torch.device) -> MyModelV4:
    model = MyModelV4(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(dict(checkpoint)["model_state_dict"])
    model.eval()
    return model


def compute_analysis_rows(
    model: MyModelV4,
    device: torch.device,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    ratings: np.ndarray,
    review_emb: Optional[np.ndarray],
    batch_size: int,
) -> pd.DataFrame:
    if review_emb is None:
        dataset = TensorDataset(
            torch.from_numpy(user_ids).long(),
            torch.from_numpy(item_ids).long(),
            torch.from_numpy(ratings).float(),
            torch.arange(len(ratings)).long(),
        )
    else:
        dataset = TensorDataset(
            torch.from_numpy(user_ids).long(),
            torch.from_numpy(item_ids).long(),
            torch.from_numpy(ratings).float(),
            torch.from_numpy(review_emb).float(),
            torch.arange(len(ratings)).long(),
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    rows = []
    with torch.no_grad():
        for batch in loader:
            if review_emb is None:
                user_id, item_id, rating, test_id = batch
                batch_review_emb = None
            else:
                user_id, item_id, rating, batch_review_emb, test_id = batch
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            rating = rating.to(device)
            if batch_review_emb is not None:
                batch_review_emb = batch_review_emb.to(device)

            outputs = model(
                user_id=user_id,
                item_id=item_id,
                review_emb=batch_review_emb,
                return_dict=True,
            )
            pred = outputs["rating_pred"]
            shared_norm = outputs["review_shared"].norm(dim=-1)
            orthogonal_norm = outputs["review_orthogonal"].norm(dim=-1)
            review_norm = outputs["review_latent"].norm(dim=-1)
            residual_ratio = orthogonal_norm / (shared_norm + orthogonal_norm + model.orthogonal_eps)
            orthogonal_to_review_ratio = orthogonal_norm / (review_norm + model.orthogonal_eps)
            shared_ratio = shared_norm / (shared_norm + orthogonal_norm + model.orthogonal_eps)
            cf_review_cos = F.cosine_similarity(
                outputs["cf_latent"],
                outputs["review_latent"],
                dim=-1,
                eps=model.orthogonal_eps,
            )

            for idx in range(pred.numel()):
                target = float(rating[idx].detach().cpu())
                prediction = float(pred[idx].detach().cpu())
                rows.append(
                    {
                        "test_id": int(test_id[idx]),
                        "user_id": int(user_id[idx].detach().cpu()),
                        "item_id": int(item_id[idx].detach().cpu()),
                        "rating": target,
                        "prediction": prediction,
                        "error": prediction - target,
                        "abs_error": abs(prediction - target),
                        "residual_ratio": float(residual_ratio[idx].detach().cpu()),
                        "orthogonal_to_review_ratio": float(orthogonal_to_review_ratio[idx].detach().cpu()),
                        "shared_ratio": float(shared_ratio[idx].detach().cpu()),
                        "shared_norm": float(shared_norm[idx].detach().cpu()),
                        "orthogonal_norm": float(orthogonal_norm[idx].detach().cpu()),
                        "review_norm": float(review_norm[idx].detach().cpu()),
                        "cf_review_cos": float(cf_review_cos[idx].detach().cpu()),
                        "review_valid": bool(outputs["review_valid_mask"][idx].detach().cpu()),
                    }
                )

    return pd.DataFrame(rows)


def select_examples(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    selections = []
    sort_specs = [
        ("high_abs_error", "abs_error", False),
        ("low_abs_error", "abs_error", True),
        ("high_residual_ratio", "residual_ratio", False),
        ("low_residual_ratio", "residual_ratio", True),
    ]

    for group in ["consistent", "inconsistent"]:
        group_df = df[df["sentiment_group"] == group]
        for label, column, ascending in sort_specs:
            part = group_df.sort_values(column, ascending=ascending).head(top_k).copy()
            part.insert(0, "selection", label)
            selections.append(part)

    if not selections:
        return df.head(0)
    return pd.concat(selections, ignore_index=True)


def write_summary(output_path: Path, full_df: pd.DataFrame) -> None:
    summary = {}
    for group, group_df in full_df.groupby("sentiment_group"):
        preds = torch.tensor(group_df["prediction"].to_numpy(dtype=np.float32))
        targets = torch.tensor(group_df["rating"].to_numpy(dtype=np.float32))
        metrics = compute_all_metrics(preds, targets)
        summary[group] = {
            "num_samples": int(len(group_df)),
            "mean_abs_error": float(group_df["abs_error"].mean()),
            "mean_residual_ratio": float(group_df["residual_ratio"].mean()),
            "mean_cf_review_cos": float(group_df["cf_review_cos"].mean()),
            **{key: float(value) for key, value in metrics.items()},
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    device = get_device(args.device)

    dataset_root = Path(args.data_root) / args.dataset
    data_dir = dataset_root / "bert"
    user_ids, item_ids, ratings, review_emb, resolved_review_source = load_test_arrays(
        data_dir,
        review_source=args.review_source,
    )

    subset_ids = load_subset_ids(dataset_root, args.subset)
    text_df = load_text_frame(dataset_root, args.subset)

    model = load_model(cfg, args.checkpoint, device)
    analysis_df = compute_analysis_rows(
        model=model,
        device=device,
        user_ids=user_ids,
        item_ids=item_ids,
        ratings=ratings,
        review_emb=review_emb,
        batch_size=args.batch_size,
    )

    group_frames = []
    for group, ids in subset_ids.items():
        group_df = analysis_df[analysis_df["test_id"].isin(set(int(x) for x in ids))].copy()
        group_df["sentiment_group"] = group
        group_frames.append(group_df)
    full_df = pd.concat(group_frames, ignore_index=True)
    full_df["review_source"] = resolved_review_source

    full_df = full_df.merge(text_df, on="test_id", how="left", suffixes=("", "_text_source"))

    output_path = Path(args.output) if args.output else Path("outputs") / "analysis" / f"{args.dataset}_{args.subset}_mymodel_v4_examples.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_df = select_examples(full_df, args.top_k)
    selected_df.to_csv(output_path, index=False)

    full_path = output_path.with_name(output_path.stem + "_all.csv")
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    full_df.to_csv(full_path, index=False)
    write_summary(summary_path, full_df)

    print(f"Saved selected examples: {output_path}")
    print(f"Saved all subset rows:    {full_path}")
    print(f"Saved summary:            {summary_path}")


if __name__ == "__main__":
    main()
