from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path

from retail_common import (
    CROPPED_DIR,
    DATA_ROOT,
    METADATA_DIR,
    RAW_DIR,
    SEGMENTATION_OVERLAY_DIR,
    SEGMENTED_DIR,
    ensure_class_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive the active retail-image dataset and recreate clean active folders."
    )
    parser.add_argument(
        "--name",
        default=f"{date.today().isoformat()}_tops_pilot",
        help="Archive folder name under data/retail_images/archive/.",
    )
    parser.add_argument("--keep-cropped-dir", action="store_true")
    return parser.parse_args()


def unique_archive_dir(base: Path) -> Path:
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = base.with_name(f"{base.name}_{index}")
        if not candidate.exists():
            return candidate
        index += 1


def move_if_present(source: Path, archive_root: Path) -> None:
    if not source.exists():
        return
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / source.name
    shutil.move(str(source), str(destination))
    print(f"Archived {source} -> {destination}")


def main() -> None:
    args = parse_args()
    archive_root = unique_archive_dir(DATA_ROOT / "archive" / args.name)
    archive_root.mkdir(parents=True, exist_ok=True)

    move_if_present(RAW_DIR, archive_root)
    if not args.keep_cropped_dir:
        move_if_present(CROPPED_DIR, archive_root)
    move_if_present(SEGMENTED_DIR, archive_root)
    move_if_present(SEGMENTATION_OVERLAY_DIR, archive_root)
    move_if_present(METADATA_DIR, archive_root)

    ensure_class_dirs(RAW_DIR)
    ensure_class_dirs(SEGMENTED_DIR)
    ensure_class_dirs(SEGMENTATION_OVERLAY_DIR)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.keep_cropped_dir:
        CROPPED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Active retail dataset folders recreated under {DATA_ROOT}")
    print(f"Archive written to {archive_root}")


if __name__ == "__main__":
    main()
