import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _save_split(common_dir: Path, bert_dir: Path, split: str) -> int:
    csv_path = common_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing common split: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"user_id", "item_id", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    np.save(bert_dir / f"{split}_user_id.npy", df["user_id"].astype(np.int64).to_numpy())
    np.save(bert_dir / f"{split}_item_id.npy", df["item_id"].astype(np.int64).to_numpy())
    np.save(bert_dir / f"{split}_rating.npy", df["rating"].astype(np.float32).to_numpy())
    return int(len(df))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync CFARG BERT split id/rating arrays from dataset/<name>/common/*.csv."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default="./dataset")
    args = parser.parse_args()

    dataset_dir = Path(args.data_root) / args.dataset
    common_dir = dataset_dir / "common"
    bert_dir = dataset_dir / "bert"
    if not common_dir.exists():
        raise FileNotFoundError(f"Missing common dir: {common_dir}")
    if not bert_dir.exists():
        raise FileNotFoundError(f"Missing bert dir: {bert_dir}")

    old_sizes = {}
    for split in ["train", "valid", "test"]:
        path = bert_dir / f"{split}_user_id.npy"
        old_sizes[split] = int(len(np.load(path))) if path.exists() else None

    new_sizes = {split: _save_split(common_dir, bert_dir, split) for split in ["train", "valid", "test"]}

    review_path = bert_dir / "review_emb.npy"
    review_size = int(np.load(review_path, mmap_mode="r").shape[0]) if review_path.exists() else None
    print(f"Synced {args.dataset} BERT split arrays from common CSVs.")
    print(f"Old sizes: {old_sizes}")
    print(f"New sizes: {new_sizes}")
    print(f"review_emb.npy rows: {review_size}")
    if review_size != new_sizes["train"]:
        print(
            "WARNING: review_emb.npy does not align with the synced train split. "
            "Use data.exclude_target_review=false for this quick run, or rerun BERT preprocessing "
            "to make leave-one-out target-review exclusion leakage-safe."
        )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
