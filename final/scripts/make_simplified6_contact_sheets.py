from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from retail_common import METADATA_DIR, RESULTS_DIR, SIMPLIFIED6_CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make contact sheets for manually curating simplified 6-class retailer images."
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=METADATA_DIR / "splits_raw_full_simplified6.csv",
    )
    parser.add_argument(
        "--classification-report",
        type=Path,
        help="Optional classification_report.csv. If provided, the worst classes by F1 are selected.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=SIMPLIFIED6_CLASS_NAMES,
        help="Simplified classes to include. Overrides --classification-report class selection.",
    )
    parser.add_argument("--worst-count", type=int, default=3)
    parser.add_argument("--examples-per-class", type=int, default=48)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--thumb-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "simplified6_contact_sheets",
    )
    return parser.parse_args()


def classes_from_report(path: Path, worst_count: int) -> list[str]:
    report = pd.read_csv(path)
    label_col = report.columns[0]
    class_rows = report[report[label_col].isin(SIMPLIFIED6_CLASS_NAMES)].copy()
    if class_rows.empty:
        raise ValueError(f"No simplified classes found in {path}")
    return class_rows.sort_values("f1-score")[label_col].head(worst_count).tolist()


def draw_contact_sheet(
    rows: pd.DataFrame,
    class_name: str,
    output_path: Path,
    columns: int,
    thumb_size: int,
) -> None:
    rows = rows.reset_index(drop=True)
    label_height = 44
    rows_count = max(1, (len(rows) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * thumb_size, rows_count * (thumb_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)

    for index, row in rows.iterrows():
        col = index % columns
        row_idx = index // columns
        x = col * thumb_size
        y = row_idx * (thumb_size + label_height)
        image_path = Path(row["path"])
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = ImageOps.contain(image, (thumb_size, thumb_size), Image.Resampling.LANCZOS)
            paste_x = x + (thumb_size - image.width) // 2
            paste_y = y + (thumb_size - image.height) // 2
            sheet.paste(image, (paste_x, paste_y))
        except OSError as exc:
            draw.text((x + 4, y + 20), f"unreadable:{exc.__class__.__name__}", fill="red")

        source_label = str(row.get("fashion_mnist_class", ""))
        file_label = image_path.name[:22]
        draw.text((x + 4, y + thumb_size + 4), source_label, fill="black")
        draw.text((x + 4, y + thumb_size + 22), file_label, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    if not args.splits.exists():
        raise FileNotFoundError(f"Split metadata does not exist: {args.splits}")
    frame = pd.read_csv(args.splits)
    required = {"path", "fashion_mnist_class", "simplified_class"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{args.splits} is missing columns: {sorted(missing)}")

    if args.classes:
        selected_classes = args.classes
    elif args.classification_report:
        selected_classes = classes_from_report(args.classification_report, args.worst_count)
    else:
        selected_classes = SIMPLIFIED6_CLASS_NAMES

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in selected_classes:
        class_rows = frame[frame["simplified_class"] == class_name]
        if class_rows.empty:
            continue
        sample_size = min(args.examples_per_class, len(class_rows))
        sample = class_rows.sample(n=sample_size, random_state=args.seed)
        output_path = args.output_dir / f"{class_name}_contact_sheet.jpg"
        draw_contact_sheet(sample, class_name, output_path, args.columns, args.thumb_size)
        print(f"{class_name}: {sample_size} examples -> {output_path}")


if __name__ == "__main__":
    main()
