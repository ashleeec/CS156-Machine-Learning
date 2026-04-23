from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "retail_images"
RAW_DIR = DATA_ROOT / "raw"
CROPPED_DIR = DATA_ROOT / "cropped"
SEGMENTED_DIR = DATA_ROOT / "segmented"
SEGMENTATION_OVERLAY_DIR = DATA_ROOT / "segmentation_overlays"
METADATA_DIR = DATA_ROOT / "metadata"
RESULTS_DIR = PROJECT_ROOT / "results" / "retail_finetune"
REPORT_DIR = PROJECT_ROOT / "report"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

SIMPLIFIED6_CLASS_NAMES = [
    "top",
    "trouser",
    "dress",
    "outerwear",
    "footwear",
    "bag",
]

CLASS_TO_SIMPLIFIED6 = {
    "T-shirt/top": "top",
    "Shirt": "top",
    "Pullover": "top",
    "Trouser": "trouser",
    "Dress": "dress",
    "Coat": "outerwear",
    "Sandal": "footwear",
    "Sneaker": "footwear",
    "Ankle boot": "footwear",
    "Bag": "bag",
}

CLASS_TO_SLUG = {
    "T-shirt/top": "t_shirt_top",
    "Trouser": "trouser",
    "Pullover": "pullover",
    "Dress": "dress",
    "Coat": "coat",
    "Sandal": "sandal",
    "Shirt": "shirt",
    "Sneaker": "sneaker",
    "Bag": "bag",
    "Ankle boot": "ankle_boot",
}
SLUG_TO_CLASS = {value: key for key, value in CLASS_TO_SLUG.items()}


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("t-shirt", "t_shirt")
    value = value.replace("/", "_")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def normalize_class_name(value: str) -> str:
    raw_value = str(value).strip()
    if raw_value in CLASS_TO_SLUG:
        return raw_value

    slug = slugify(raw_value)
    if slug in SLUG_TO_CLASS:
        return SLUG_TO_CLASS[slug]

    valid = ", ".join(CLASS_NAMES)
    raise ValueError(f"Unknown Fashion-MNIST class {value!r}. Valid classes: {valid}")


def simplify6_class_name(value: str) -> str:
    class_name = normalize_class_name(value)
    return CLASS_TO_SIMPLIFIED6[class_name]


def class_dir(root: Path, class_name: str) -> Path:
    return root / CLASS_TO_SLUG[normalize_class_name(class_name)]


def ensure_class_dirs(root: Path = RAW_DIR) -> None:
    for class_name in CLASS_NAMES:
        class_dir(root, class_name).mkdir(parents=True, exist_ok=True)


def image_extensions() -> tuple[str, ...]:
    return (".jpg", ".jpeg", ".png", ".webp", ".bmp")
