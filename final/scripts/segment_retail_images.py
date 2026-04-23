from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from retail_common import (
    CLASS_NAMES,
    RAW_DIR,
    SEGMENTATION_OVERLAY_DIR,
    SEGMENTED_DIR,
    class_dir,
    ensure_class_dirs,
    image_extensions,
)


DEFAULT_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
METADATA_COLUMNS = [
    "source_path",
    "segmented_path",
    "overlay_path",
    "fashion_mnist_class",
    "target_labels",
    "matched_labels",
    "mask_area_ratio",
    "left",
    "top",
    "right",
    "bottom",
    "status",
    "message",
    "model_id",
    "output_mode",
]

TARGET_LABELS = {
    "T-shirt/top": ["Upper-clothes"],
    "Shirt": ["Upper-clothes"],
    "Pullover": ["Upper-clothes"],
    "Dress": ["Dress"],
    "Trouser": ["Pants"],
    "Sandal": ["Left-shoe", "Right-shoe"],
    "Sneaker": ["Left-shoe", "Right-shoe"],
    "Ankle boot": ["Left-shoe", "Right-shoe"],
    "Bag": ["Bag"],
    # The default ATR-style clothes parser does not expose Coat. This keeps
    # coats usable while flagging them in metadata as a fallback label choice.
    "Coat": ["Coat", "Upper-clothes"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment retailer images with a clothing parser and save model-ready crops."
    )
    parser.add_argument("--input-root", type=Path, default=RAW_DIR)
    parser.add_argument("--output-root", type=Path, default=SEGMENTED_DIR)
    parser.add_argument("--overlay-root", type=Path, default=SEGMENTATION_OVERLAY_DIR)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/retail_images/metadata/segmentations.csv"),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--output-mode",
        choices=["crop_masked", "crop", "full_masked", "copy"],
        default="crop_masked",
    )
    parser.add_argument(
        "--fallback",
        choices=["skip", "copy"],
        default="skip",
        help="What to do when the target clothing mask is missing or unusable.",
    )
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.01)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.90)
    parser.add_argument("--min-side", type=int, default=96)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install it with:\n"
            "  python3 -m pip install --user transformers huggingface_hub safetensors\n"
            "Then rerun this script."
        ) from exc
    return AutoModelForSemanticSegmentation, SegformerImageProcessor


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_label(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum())


def model_label_lookup(model: torch.nn.Module) -> dict[str, int]:
    id2label = getattr(model.config, "id2label", {}) or {}
    return {normalize_label(label): int(idx) for idx, label in id2label.items()}


def target_label_ids(class_name: str, lookup: dict[str, int]) -> tuple[list[int], list[str]]:
    labels = TARGET_LABELS[class_name]
    ids: list[int] = []
    matched: list[str] = []
    for label in labels:
        normalized = normalize_label(label)
        if normalized in lookup:
            ids.append(lookup[normalized])
            matched.append(label)
    return ids, matched


def iter_images(root: Path, limit_per_class: int | None) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for class_name in CLASS_NAMES:
        folder = class_dir(root, class_name)
        if not folder.exists():
            continue
        count = 0
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in image_extensions():
                continue
            rows.append((class_name, path))
            count += 1
            if limit_per_class is not None and count >= limit_per_class:
                break
    return rows


def padded_bbox(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    box_width = right - left + 1
    box_height = bottom - top + 1
    x_pad = int(round(box_width * padding))
    y_pad = int(round(box_height * padding))
    return (
        max(0, left - x_pad),
        max(0, top - y_pad),
        min(width, right + 1 + x_pad),
        min(height, bottom + 1 + y_pad),
    )


def mask_bbox(mask: np.ndarray, padding: float) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    height, width = mask.shape
    return padded_bbox(
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
        width,
        height,
        padding,
    )


def make_segmented_image(
    image: Image.Image,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    mode: str,
) -> Image.Image:
    image = image.convert("RGB")
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    white = Image.new("RGB", image.size, "white")

    if mode == "copy":
        return image.copy()
    if mode == "full_masked":
        return Image.composite(image, white, mask_image)

    left, top, right, bottom = bbox
    if mode == "crop":
        return image.crop((left, top, right, bottom))

    masked = Image.composite(image, white, mask_image)
    return masked.crop((left, top, right, bottom))


def save_overlay(
    image: Image.Image,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    overlay = image.convert("RGBA")
    color = Image.new("RGBA", overlay.size, (255, 64, 64, 0))
    alpha = Image.fromarray((mask.astype(np.uint8) * 110), mode="L")
    color.putalpha(alpha)
    combined = Image.alpha_composite(overlay, color)
    draw = ImageDraw.Draw(combined)
    draw.rectangle(bbox, outline=(0, 255, 128, 255), width=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.convert("RGB").save(output_path, format="JPEG", quality=92)


def run_segmentation(
    image: Image.Image,
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    resized = F.interpolate(
        logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.argmax(dim=1)[0].detach().cpu().numpy()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in METADATA_COLUMNS})


def fallback_copy(
    image: Image.Image,
    input_path: Path,
    class_name: str,
    output_path: Path,
    overlay_path: Path,
    message: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="JPEG", quality=92)
    image.convert("RGB").save(overlay_path, format="JPEG", quality=92)
    return {
        "source_path": str(input_path),
        "segmented_path": str(output_path),
        "overlay_path": str(overlay_path),
        "fashion_mnist_class": class_name,
        "target_labels": "|".join(TARGET_LABELS[class_name]),
        "matched_labels": "",
        "mask_area_ratio": "",
        "status": "fallback_copy",
        "message": message,
        "model_id": args.model_id,
        "output_mode": "copy",
    }


def main() -> None:
    args = parse_args()
    AutoModelForSemanticSegmentation, SegformerImageProcessor = require_transformers()
    ensure_class_dirs(args.output_root)
    ensure_class_dirs(args.overlay_root)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    processor = SegformerImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_id).to(device)
    model.eval()
    lookup = model_label_lookup(model)

    rows: list[dict[str, Any]] = []
    for class_name, input_path in iter_images(args.input_root, args.limit_per_class):
        output_path = class_dir(args.output_root, class_name) / f"{input_path.stem}_seg.jpg"
        overlay_path = class_dir(args.overlay_root, class_name) / f"{input_path.stem}_overlay.jpg"
        if output_path.exists() and overlay_path.exists() and not args.overwrite:
            rows.append(
                {
                    "source_path": str(input_path),
                    "segmented_path": str(output_path),
                    "overlay_path": str(overlay_path),
                    "fashion_mnist_class": class_name,
                    "target_labels": "|".join(TARGET_LABELS[class_name]),
                    "status": "exists",
                    "message": "skipped_existing",
                    "model_id": args.model_id,
                    "output_mode": args.output_mode,
                }
            )
            continue

        try:
            with Image.open(input_path) as original:
                image = original.convert("RGB")
                target_ids, matched_labels = target_label_ids(class_name, lookup)
                if not target_ids:
                    message = f"target_labels_missing_from_model:{TARGET_LABELS[class_name]}"
                    if args.fallback == "copy":
                        rows.append(
                            fallback_copy(
                                image, input_path, class_name, output_path, overlay_path, message, args
                            )
                        )
                    else:
                        rows.append(
                            {
                                "source_path": str(input_path),
                                "fashion_mnist_class": class_name,
                                "target_labels": "|".join(TARGET_LABELS[class_name]),
                                "status": "skipped",
                                "message": message,
                                "model_id": args.model_id,
                                "output_mode": args.output_mode,
                            }
                        )
                    continue

                predicted = run_segmentation(image, processor, model, device)
                mask = np.isin(predicted, target_ids)
                mask_area_ratio = float(mask.mean())
                bbox = mask_bbox(mask, args.padding)
                if bbox is None:
                    message = "missing_target_mask"
                    if args.fallback == "copy":
                        rows.append(
                            fallback_copy(
                                image, input_path, class_name, output_path, overlay_path, message, args
                            )
                        )
                    else:
                        rows.append(
                            {
                                "source_path": str(input_path),
                                "fashion_mnist_class": class_name,
                                "target_labels": "|".join(TARGET_LABELS[class_name]),
                                "matched_labels": "|".join(matched_labels),
                                "mask_area_ratio": mask_area_ratio,
                                "status": "skipped",
                                "message": message,
                                "model_id": args.model_id,
                                "output_mode": args.output_mode,
                            }
                        )
                    continue

                left, top, right, bottom = bbox
                segmented = make_segmented_image(image, mask, bbox, args.output_mode)
                if min(segmented.size) < args.min_side:
                    message = f"segmented_too_small:{segmented.size[0]}x{segmented.size[1]}"
                    if args.fallback == "copy":
                        rows.append(
                            fallback_copy(
                                image, input_path, class_name, output_path, overlay_path, message, args
                            )
                        )
                        continue
                    rows.append(
                        {
                            "source_path": str(input_path),
                            "fashion_mnist_class": class_name,
                            "target_labels": "|".join(TARGET_LABELS[class_name]),
                            "matched_labels": "|".join(matched_labels),
                            "mask_area_ratio": mask_area_ratio,
                            "left": left,
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                            "status": "skipped",
                            "message": message,
                            "model_id": args.model_id,
                            "output_mode": args.output_mode,
                        }
                    )
                    continue
                if not (args.min_mask_area_ratio <= mask_area_ratio <= args.max_mask_area_ratio):
                    message = f"mask_area_out_of_range:{mask_area_ratio:.4f}"
                    if args.fallback == "copy":
                        rows.append(
                            fallback_copy(
                                image, input_path, class_name, output_path, overlay_path, message, args
                            )
                        )
                    else:
                        rows.append(
                            {
                                "source_path": str(input_path),
                                "fashion_mnist_class": class_name,
                                "target_labels": "|".join(TARGET_LABELS[class_name]),
                                "matched_labels": "|".join(matched_labels),
                                "mask_area_ratio": mask_area_ratio,
                                "left": left,
                                "top": top,
                                "right": right,
                                "bottom": bottom,
                                "status": "skipped",
                                "message": message,
                                "model_id": args.model_id,
                                "output_mode": args.output_mode,
                            }
                        )
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)
                segmented.save(output_path, format="JPEG", quality=92)
                save_overlay(image, mask, bbox, overlay_path)
        except OSError as exc:
            rows.append(
                {
                    "source_path": str(input_path),
                    "fashion_mnist_class": class_name,
                    "status": "skipped",
                    "message": f"unreadable:{exc.__class__.__name__}",
                    "model_id": args.model_id,
                    "output_mode": args.output_mode,
                }
            )
            continue

        rows.append(
            {
                "source_path": str(input_path),
                "segmented_path": str(output_path),
                "overlay_path": str(overlay_path),
                "fashion_mnist_class": class_name,
                "target_labels": "|".join(TARGET_LABELS[class_name]),
                "matched_labels": "|".join(matched_labels),
                "mask_area_ratio": mask_area_ratio,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "status": "segmented",
                "message": "ok",
                "model_id": args.model_id,
                "output_mode": args.output_mode,
            }
        )

    write_rows(args.metadata, rows)
    summary = pd.DataFrame(rows)
    if summary.empty:
        print("No input images found.")
    else:
        print(summary.groupby(["fashion_mnist_class", "status"]).size().unstack(fill_value=0))
    print(f"Segmentation metadata written to {args.metadata}")
    print(f"Segmented images written to {args.output_root}")
    print(f"QA overlays written to {args.overlay_root}")


if __name__ == "__main__":
    main()
