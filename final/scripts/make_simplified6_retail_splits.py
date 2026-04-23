from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from retail_common import METADATA_DIR, SIMPLIFIED6_CLASS_NAMES, simplify6_class_name, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced 6-class raw-full retailer split from the 10-class split."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=METADATA_DIR / "splits_raw_full.csv",
        help="Raw full-image split with Fashion-MNIST labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=METADATA_DIR / "splits_raw_full_simplified6.csv",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("data/retail_images/simplified6_raw_full"),
        help="Stable folder where selected raw full-image files are copied for training.",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Keep original raw paths instead of copying selected images to --prepared-root.",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=0,
        help="Rows per simplified class. Use 0 to auto-balance to the smallest available class.",
    )
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input split does not exist: {path}")
    frame = pd.read_csv(path)
    required = {"path", "fashion_mnist_class"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["simplified_class"] = frame["fashion_mnist_class"].map(simplify6_class_name)
    exists = frame["path"].map(lambda value: Path(value).exists())
    missing_count = int((~exists).sum())
    if missing_count:
        print(f"Dropping {missing_count} rows with missing image files before balancing.")
        frame = frame[exists].copy()
    segmented_paths = frame["path"].astype(str).str.contains("/segmented/|data/retail_images/segmented")
    if segmented_paths.any():
        raise ValueError("Simplified raw-full split must not include segmented image paths.")
    return frame


def balanced_subset(frame: pd.DataFrame, target_per_class: int, seed: int) -> pd.DataFrame:
    pieces = []
    too_small: dict[str, int] = {}
    if target_per_class <= 0:
        counts = frame["simplified_class"].value_counts()
        target_per_class = int(counts.reindex(SIMPLIFIED6_CLASS_NAMES).min())
        print(f"Auto-selected target_per_class={target_per_class}.")
    for class_name in SIMPLIFIED6_CLASS_NAMES:
        class_rows = frame[frame["simplified_class"] == class_name]
        if len(class_rows) < target_per_class:
            too_small[class_name] = len(class_rows)
            continue
        pieces.append(class_rows.sample(n=target_per_class, random_state=seed))
    if too_small:
        raise ValueError(
            f"Not enough rows for target_per_class={target_per_class}: {too_small}"
        )
    return pd.concat(pieces, ignore_index=True)


def materialize_images(frame: pd.DataFrame, prepared_root: Path) -> pd.DataFrame:
    frame = frame.copy()
    materialized_paths = []
    for row in frame.itertuples(index=False):
        source_path = Path(row.path)
        if not source_path.exists():
            raise FileNotFoundError(f"Selected image disappeared before copy: {source_path}")
        class_dir = prepared_root / str(row.simplified_class)
        class_dir.mkdir(parents=True, exist_ok=True)
        original_slug = slugify(str(row.fashion_mnist_class))
        output_path = class_dir / f"{original_slug}_{source_path.name}"
        if not output_path.exists():
            shutil.copy2(source_path, output_path)
        materialized_paths.append(str(output_path))
    frame["source_path"] = frame["path"]
    frame["path"] = materialized_paths
    return frame


def create_splits(
    frame: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> pd.DataFrame:
    counts = frame["simplified_class"].value_counts()
    represented_classes = counts.index.nunique()
    too_small = counts[counts < 3]
    if not too_small.empty:
        raise ValueError(
            "Need at least 3 examples per simplified class for train/val/test splits. "
            f"Too small: {too_small.to_dict()}"
        )

    test_count = math.ceil(len(frame) * test_size)
    train_val_count = len(frame) - test_count
    val_count = math.ceil(train_val_count * (val_size / (1.0 - test_size)))
    if test_count < represented_classes or val_count < represented_classes:
        raise ValueError(
            "Dataset is too small for stratified train/val/test splits with the requested ratios. "
            f"Rows={len(frame)}, classes={represented_classes}, test rows={test_count}, val rows={val_count}."
        )

    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        stratify=frame["simplified_class"],
        random_state=seed,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val["simplified_class"],
        random_state=seed,
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    return pd.concat([train, val, test], ignore_index=True)


def main() -> None:
    args = parse_args()
    source = load_source(args.input)
    balanced = balanced_subset(source, args.target_per_class, args.seed)
    if not args.no_copy:
        balanced = materialize_images(balanced, args.prepared_root)
    splits = create_splits(balanced, args.val_size, args.test_size, args.seed)
    splits = splits.sort_values(["split", "simplified_class", "path"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.output, index=False)

    print(pd.crosstab(splits["simplified_class"], splits["split"]).reindex(SIMPLIFIED6_CLASS_NAMES))
    print("Original labels per simplified class:")
    print(pd.crosstab(splits["simplified_class"], splits["fashion_mnist_class"]).reindex(SIMPLIFIED6_CLASS_NAMES))
    print(f"Simplified split written to {args.output}")


if __name__ == "__main__":
    main()
