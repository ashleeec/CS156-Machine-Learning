from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from retail_common import (
    CLASS_NAMES,
    CLASS_TO_SLUG,
    METADATA_DIR,
    SEGMENTED_DIR,
    class_dir,
    ensure_class_dirs,
    image_extensions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate downloaded retailer images and create stratified splits."
    )
    parser.add_argument("--raw-root", type=Path, default=SEGMENTED_DIR)
    parser.add_argument("--output", type=Path, default=METADATA_DIR / "validated_images.csv")
    parser.add_argument("--splits", type=Path, default=METADATA_DIR / "splits.csv")
    parser.add_argument("--min-side", type=int, default=160)
    parser.add_argument("--target-per-class", type=int, default=150)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-small",
        action="store_true",
        help="Write validation output even if a stratified split is too small.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path, class_name: str, min_side: int) -> dict[str, object]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
    except (UnidentifiedImageError, OSError) as exc:
        return {
            "path": str(path),
            "fashion_mnist_class": class_name,
            "width": None,
            "height": None,
            "mode": None,
            "sha256": None,
            "is_valid": False,
            "validation_message": f"unreadable:{exc.__class__.__name__}",
        }

    valid_size = min(width, height) >= min_side
    return {
        "path": str(path),
        "fashion_mnist_class": class_name,
        "width": width,
        "height": height,
        "mode": mode,
        "sha256": sha256_file(path),
        "is_valid": valid_size,
        "validation_message": "ok" if valid_size else f"too_small:{width}x{height}",
    }


def collect_rows(raw_root: Path, min_side: int) -> pd.DataFrame:
    rows = []
    ensure_class_dirs(raw_root)
    for class_name in CLASS_NAMES:
        folder = class_dir(raw_root, class_name)
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in image_extensions():
                continue
            rows.append(inspect_image(path, class_name, min_side))
    columns = [
        "path",
        "fashion_mnist_class",
        "width",
        "height",
        "mode",
        "sha256",
        "is_valid",
        "validation_message",
    ]
    return pd.DataFrame(rows, columns=columns)


def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["is_duplicate"] = []
        df["usable"] = []
        return df
    df = df.copy()
    df["is_duplicate"] = False
    valid_hashes = df["is_valid"] & df["sha256"].notna()
    df.loc[valid_hashes, "is_duplicate"] = df.loc[valid_hashes, "sha256"].duplicated()
    df["usable"] = df["is_valid"] & ~df["is_duplicate"]
    return df


def balanced_subset(df: pd.DataFrame, target_per_class: int, seed: int) -> pd.DataFrame:
    usable = df[df["usable"]].copy()
    pieces = []
    for class_name in CLASS_NAMES:
        class_rows = usable[usable["fashion_mnist_class"] == class_name]
        if len(class_rows) > target_per_class:
            class_rows = class_rows.sample(n=target_per_class, random_state=seed)
        pieces.append(class_rows)
    return pd.concat(pieces, ignore_index=True) if pieces else usable


def create_splits(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
    allow_small: bool,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No usable images were found.")
    counts = df["fashion_mnist_class"].value_counts()
    represented_classes = counts.index.nunique()
    too_small = counts[counts < 3]
    if not too_small.empty:
        message = (
            "Need at least 3 usable images per represented class for train/val/test splits. "
            f"Too small: {too_small.to_dict()}"
        )
        if allow_small:
            output = df.copy()
            output["split"] = "train"
            return output
        raise ValueError(message)

    test_count = math.ceil(len(df) * test_size)
    train_val_count = len(df) - test_count
    val_count = math.ceil(train_val_count * (val_size / (1.0 - test_size)))
    if test_count < represented_classes or val_count < represented_classes:
        message = (
            "Dataset is too small for stratified train/val/test splits with the requested ratios. "
            f"Rows={len(df)}, classes={represented_classes}, test rows={test_count}, val rows={val_count}."
        )
        if allow_small:
            output = df.copy()
            output["split"] = "train"
            return output
        raise ValueError(message)

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["fashion_mnist_class"],
        random_state=seed,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val["fashion_mnist_class"],
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
    rows = collect_rows(args.raw_root, args.min_side)
    rows = mark_duplicates(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output, index=False)

    usable_balanced = balanced_subset(rows, args.target_per_class, args.seed)
    splits = create_splits(
        usable_balanced,
        args.val_size,
        args.test_size,
        args.seed,
        args.allow_small,
    )
    args.splits.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.splits, index=False)

    summary = (
        rows.groupby("fashion_mnist_class")
        .agg(total=("path", "count"), usable=("usable", "sum"), duplicates=("is_duplicate", "sum"))
        .reindex(CLASS_NAMES, fill_value=0)
    )
    print(summary)
    print(f"Validation metadata written to {args.output}")
    print(f"Stratified splits written to {args.splits}")


if __name__ == "__main__":
    main()
