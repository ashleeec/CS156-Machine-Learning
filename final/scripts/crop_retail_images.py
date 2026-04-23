from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import models
from torchvision.transforms import functional as TVF

from retail_common import (
    CLASS_NAMES,
    CROPPED_DIR,
    RAW_DIR,
    class_dir,
    ensure_class_dirs,
    image_extensions,
)


CLASS_REGIONS = {
    "T-shirt/top": (0.10, 0.18, 0.90, 0.68),
    "Shirt": (0.10, 0.18, 0.90, 0.70),
    "Pullover": (0.08, 0.16, 0.92, 0.72),
    "Coat": (0.06, 0.14, 0.94, 0.86),
    "Dress": (0.08, 0.12, 0.92, 0.95),
    "Trouser": (0.16, 0.42, 0.84, 1.00),
    "Sandal": (0.06, 0.74, 0.94, 1.00),
    "Sneaker": (0.06, 0.74, 0.94, 1.00),
    "Ankle boot": (0.06, 0.70, 0.94, 1.00),
    "Bag": (0.08, 0.12, 0.92, 0.88),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop raw retailer images into garment-focused training images."
    )
    parser.add_argument("--input-root", type=Path, default=RAW_DIR)
    parser.add_argument("--output-root", type=Path, default=CROPPED_DIR)
    parser.add_argument("--metadata", type=Path, default=Path("data/retail_images/metadata/crops.csv"))
    parser.add_argument(
        "--mode",
        choices=["copy", "foreground", "heuristic", "person"],
        default="foreground",
        help=(
            "copy preserves the full image; foreground crops the main non-background area; "
            "heuristic applies class-specific crop windows without person detection; "
            "person first detects a person and then crops the relevant garment region."
        ),
    )
    parser.add_argument("--person-threshold", type=float, default=0.60)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--min-side", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_person_detector(device: torch.device) -> torch.nn.Module:
    weights = models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.eval()
    return model.to(device)


def clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    left = max(0, min(width - 1, int(round(left))))
    top = max(0, min(height - 1, int(round(top))))
    right = max(left + 1, min(width, int(round(right))))
    bottom = max(top + 1, min(height, int(round(bottom))))
    return left, top, right, bottom


def padded_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top
    return clamp_box(
        (
            left - box_width * padding,
            top - box_height * padding,
            right + box_width * padding,
            bottom + box_height * padding,
        ),
        width,
        height,
    )


def normalized_region_box(
    base_box: tuple[int, int, int, int],
    region: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    left, top, right, bottom = base_box
    base_width = right - left
    base_height = bottom - top
    rel_left, rel_top, rel_right, rel_bottom = region
    return (
        left + base_width * rel_left,
        top + base_height * rel_top,
        left + base_width * rel_right,
        top + base_height * rel_bottom,
    )


def detect_person_box(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
) -> tuple[int, int, int, int] | None:
    tensor = TVF.to_tensor(image).to(device)
    with torch.no_grad():
        prediction = model([tensor])[0]
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    boxes = prediction["boxes"].detach().cpu().numpy()
    candidates = [
        (float(score), box)
        for label, score, box in zip(labels, scores, boxes)
        if int(label) == 1 and float(score) >= threshold
    ]
    if not candidates:
        return None
    _, best_box = max(
        candidates,
        key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
    )
    width, height = image.size
    return clamp_box(tuple(float(value) for value in best_box), width, height)


def foreground_box(image: Image.Image) -> tuple[int, int, int, int] | None:
    grayscale = ImageOps.grayscale(image)
    width, height = grayscale.size
    pixels = grayscale.load()
    values = list(grayscale.resize((64, 64)).getdata())
    median = sorted(values)[len(values) // 2]
    threshold = 24
    xs: list[int] = []
    ys: list[int] = []
    step_x = max(1, width // 220)
    step_y = max(1, height // 220)
    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            if abs(int(pixels[x, y]) - median) > threshold:
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return None
    return clamp_box((min(xs), min(ys), max(xs), max(ys)), width, height)


def crop_box_for_image(
    image: Image.Image,
    class_name: str,
    mode: str,
    padding: float,
    detector: torch.nn.Module | None,
    device: torch.device | None,
    person_threshold: float,
) -> tuple[int, int, int, int, str]:
    width, height = image.size
    full_box = (0, 0, width, height)
    base_box = full_box
    method = "full_image_copy"

    if mode == "copy":
        return (*full_box, method)

    if mode == "foreground":
        foreground = foreground_box(image)
        if foreground is not None:
            return (*padded_box(foreground, width, height, padding), "foreground")
        return (*full_box, "foreground_failed_full_image")

    if mode == "person" and detector is not None and device is not None:
        detected = detect_person_box(image, detector, device, person_threshold)
        if detected is not None:
            base_box = detected
            method = "person_detector_class_region"
        else:
            foreground = foreground_box(image)
            if foreground is not None:
                return (*padded_box(foreground, width, height, padding), "person_failed_foreground")
            return (*full_box, "person_failed_full_image")

    elif mode == "heuristic":
        foreground = foreground_box(image)
        if foreground is not None:
            base_box = foreground
            method = "foreground_class_region"

    region = CLASS_REGIONS[class_name]
    region_box = normalized_region_box(base_box, region)
    return (*padded_box(region_box, width, height, padding), method)


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_images(root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for class_name in CLASS_NAMES:
        folder = class_dir(root, class_name)
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in image_extensions():
                rows.append((class_name, path))
    return rows


def main() -> None:
    args = parse_args()
    ensure_class_dirs(args.output_root)
    device = run_device() if args.mode == "person" else None
    detector = load_person_detector(device) if args.mode == "person" and device is not None else None

    rows = []
    for class_name, input_path in iter_images(args.input_root):
        output_path = class_dir(args.output_root, class_name) / f"{input_path.stem}_crop.jpg"
        if output_path.exists() and not args.overwrite:
            rows.append(
                {
                    "source_path": str(input_path),
                    "cropped_path": str(output_path),
                    "fashion_mnist_class": class_name,
                    "status": "exists",
                    "message": "skipped_existing",
                }
            )
            continue

        try:
            with Image.open(input_path) as original:
                image = original.convert("RGB")
                left, top, right, bottom, method = crop_box_for_image(
                    image,
                    class_name,
                    args.mode,
                    args.padding,
                    detector,
                    device,
                    args.person_threshold,
                )
                crop = image.crop((left, top, right, bottom))
                if min(crop.size) < args.min_side:
                    rows.append(
                        {
                            "source_path": str(input_path),
                            "cropped_path": "",
                            "fashion_mnist_class": class_name,
                            "status": "skipped",
                            "message": f"crop_too_small:{crop.size[0]}x{crop.size[1]}",
                            "crop_method": method,
                            "left": left,
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                        }
                    )
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(output_path, format="JPEG", quality=92)
        except OSError as exc:
            rows.append(
                {
                    "source_path": str(input_path),
                    "cropped_path": "",
                    "fashion_mnist_class": class_name,
                    "status": "skipped",
                    "message": f"unreadable:{exc.__class__.__name__}",
                }
            )
            continue

        rows.append(
            {
                "source_path": str(input_path),
                "cropped_path": str(output_path),
                "fashion_mnist_class": class_name,
                "status": "cropped",
                "message": "ok",
                "crop_method": method,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "source_sha256": image_hash(input_path),
                "crop_sha256": image_hash(output_path),
            }
        )

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.metadata, index=False)
    summary = pd.DataFrame(rows)
    if summary.empty:
        print("No images found to crop.")
    else:
        print(summary.groupby(["fashion_mnist_class", "status"]).size().unstack(fill_value=0))
    print(f"Crop metadata written to {args.metadata}")
    print(f"Cropped images written to {args.output_root}")


if __name__ == "__main__":
    main()
