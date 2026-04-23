from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retail_common import CLASS_NAMES, METADATA_DIR
from validate_retail_dataset import create_splits


FOOTWEAR_CLASSES = {"Sandal", "Sneaker", "Ankle boot"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a real-image split that avoids destructive footwear segmentation: "
            "raw full images for footwear, successful segmentation crops for other classes."
        )
    )
    parser.add_argument(
        "--raw-validation",
        type=Path,
        default=METADATA_DIR / "validated_raw_images.csv",
        help="Validation metadata for data/retail_images/raw.",
    )
    parser.add_argument(
        "--segmented-validation",
        type=Path,
        default=METADATA_DIR / "validated_segmented_only.csv",
        help="Validation metadata for successful segmentation crops.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=METADATA_DIR / "validated_hybrid_raw_footwear.csv",
    )
    parser.add_argument(
        "--splits-output",
        type=Path,
        default=METADATA_DIR / "splits_hybrid_raw_footwear.csv",
    )
    parser.add_argument("--target-per-class", type=int, default=100)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_usable(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required metadata does not exist: {path}")
    frame = pd.read_csv(path)
    required = {"path", "fashion_mnist_class", "usable"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame[frame["usable"]].copy()


def balanced_subset(frame: pd.DataFrame, target_per_class: int, seed: int) -> pd.DataFrame:
    pieces = []
    too_small: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        class_rows = frame[frame["fashion_mnist_class"] == class_name]
        if len(class_rows) < target_per_class:
            too_small[class_name] = len(class_rows)
            continue
        pieces.append(class_rows.sample(n=target_per_class, random_state=seed))
    if too_small:
        raise ValueError(
            f"Not enough usable rows for target_per_class={target_per_class}: {too_small}"
        )
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    args = parse_args()

    raw = load_usable(args.raw_validation)
    raw = raw[raw["fashion_mnist_class"].isin(FOOTWEAR_CLASSES)].copy()
    raw["input_policy"] = "raw_full_footwear"
    raw["source_path"] = raw["path"]
    raw["segmentation_status"] = "not_used"
    raw["mask_area_ratio"] = 0.0

    segmented = load_usable(args.segmented_validation)
    segmented = segmented[~segmented["fashion_mnist_class"].isin(FOOTWEAR_CLASSES)].copy()
    segmented["input_policy"] = "segmented_crop"

    common_columns = [
        "path",
        "source_path",
        "fashion_mnist_class",
        "width",
        "height",
        "mode",
        "sha256",
        "is_valid",
        "validation_message",
        "segmentation_status",
        "mask_area_ratio",
        "is_duplicate",
        "usable",
        "input_policy",
    ]
    combined = pd.concat(
        [raw.reindex(columns=common_columns), segmented.reindex(columns=common_columns)],
        ignore_index=True,
    )
    combined = combined.sort_values(["fashion_mnist_class", "path"]).reset_index(drop=True)

    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.validation_output, index=False)

    balanced = balanced_subset(combined, args.target_per_class, args.seed)
    splits = create_splits(
        balanced,
        args.val_size,
        args.test_size,
        args.seed,
        allow_small=False,
    )
    args.splits_output.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.splits_output, index=False)

    counts = (
        combined.groupby(["fashion_mnist_class", "input_policy"])
        .size()
        .unstack(fill_value=0)
        .reindex(CLASS_NAMES, fill_value=0)
    )
    print(counts)
    print(f"Validation metadata written to {args.validation_output}")
    print(f"Hybrid splits written to {args.splits_output}")


if __name__ == "__main__":
    main()
