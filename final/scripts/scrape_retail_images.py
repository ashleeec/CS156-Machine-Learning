from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError

from retail_common import CLASS_NAMES, RAW_DIR, class_dir, ensure_class_dirs, normalize_class_name, slugify


DEFAULT_USER_AGENT = "CS156-retail-image-research/1.0 (+polite academic dataset builder)"
IMAGE_ATTRS = ("src", "data-src", "data-original", "data-zoom-image", "data-image", "content")
METADATA_COLUMNS = [
    "downloaded_at",
    "source_name",
    "source_url",
    "fashion_mnist_class",
    "image_url",
    "filename",
    "width",
    "height",
    "sha256",
    "status",
    "message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download retailer clothing images from a source manifest."
    )
    parser.add_argument("--sources", type=Path, default=Path("configs/retailer_sources.csv"))
    parser.add_argument("--output-root", type=Path, default=RAW_DIR)
    parser.add_argument("--metadata", type=Path, default=Path("data/retail_images/metadata/downloads.csv"))
    parser.add_argument("--target-per-class", type=int, default=150)
    parser.add_argument("--max-images-per-source", type=int, default=80)
    parser.add_argument("--min-side", type=int, default=160)
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between network requests.")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--allow-missing-robots",
        action="store_true",
        help="Allow downloads when robots.txt cannot be fetched. Default is to skip.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse pages and print candidate counts without downloading images.",
    )
    return parser.parse_args()


def load_sources(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Source manifest does not exist: {path}")
    sources = pd.read_csv(path).fillna("")
    required = {"source_name", "url", "fashion_mnist_class", "notes"}
    missing = required.difference(sources.columns)
    if missing:
        raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")

    sources = sources[sources["url"].astype(str).str.strip() != ""].copy()
    if sources.empty:
        raise ValueError(
            "No source URLs found. Add retailer listing/product URLs to configs/retailer_sources.csv."
        )
    sources["fashion_mnist_class"] = sources["fashion_mnist_class"].map(normalize_class_name)
    return sources


def count_existing_images(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        folder = class_dir(root, class_name)
        if not folder.exists():
            counts[class_name] = 0
            continue
        counts[class_name] = sum(1 for path in folder.iterdir() if path.is_file())
    return counts


def read_existing_metadata(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        metadata = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return set()
    if "image_url" not in metadata.columns:
        return set()
    return set(metadata["image_url"].dropna().astype(str))


def robots_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def can_fetch(
    url: str,
    user_agent: str,
    cache: dict[str, tuple[bool, robotparser.RobotFileParser | None]],
    allow_missing_robots: bool,
    timeout: float,
) -> tuple[bool, str]:
    parsed = urlparse(url)
    key = f"{parsed.scheme}://{parsed.netloc}"
    if key not in cache:
        parser = robotparser.RobotFileParser()
        parser.set_url(robots_url(url))
        try:
            response = requests.get(
                robots_url(url),
                headers={"User-Agent": user_agent},
                timeout=timeout,
            )
            if response.status_code >= 400:
                cache[key] = (allow_missing_robots, None)
            else:
                parser.parse(response.text.splitlines())
                cache[key] = (True, parser)
        except requests.RequestException:
            cache[key] = (allow_missing_robots, None)

    robots_available, parser = cache[key]
    if parser is None:
        if robots_available:
            return True, "robots_missing_allowed"
        return False, "robots_unavailable"
    allowed = parser.can_fetch(user_agent, url)
    return allowed, "allowed" if allowed else "blocked_by_robots"


def extract_srcset(value: str, base_url: str) -> list[str]:
    urls: list[str] = []
    for candidate in value.split(","):
        piece = candidate.strip().split(" ")[0]
        if piece:
            urls.append(urljoin(base_url, piece))
    return urls


def find_images_in_json(value: Any, base_url: str) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"image", "images", "thumbnail", "thumbnailurl"}:
                urls.extend(find_images_in_json(item, base_url))
            elif isinstance(item, (dict, list)):
                urls.extend(find_images_in_json(item, base_url))
    elif isinstance(value, list):
        for item in value:
            urls.extend(find_images_in_json(item, base_url))
    elif isinstance(value, str) and value.startswith(("http://", "https://", "//", "/")):
        urls.append(urljoin(base_url, value))
    return urls


def collect_image_candidates(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []

    for tag in soup.find_all(["img", "source", "meta"]):
        for attr in IMAGE_ATTRS:
            value = tag.get(attr)
            if value:
                candidates.append(urljoin(base_url, value))
        srcset = tag.get("srcset") or tag.get("data-srcset")
        if srcset:
            candidates.extend(extract_srcset(srcset, base_url))

    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            payload = json.loads(script.string)
        except json.JSONDecodeError:
            continue
        candidates.extend(find_images_in_json(payload, base_url))

    seen: set[str] = set()
    deduped: list[str] = []
    for url in candidates:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def shopify_products_json_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}/products.json?limit=250"


def collect_shopify_image_candidates(source_url: str, user_agent: str, timeout: float) -> list[str]:
    json_url = shopify_products_json_url(source_url)
    try:
        response = requests.get(json_url, headers={"User-Agent": user_agent}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    candidates: list[str] = []
    for product in payload.get("products", []):
        if not isinstance(product, dict):
            continue
        for image in product.get("images", []):
            if isinstance(image, dict) and image.get("src"):
                candidates.append(str(image["src"]))

    deduped: list[str] = []
    seen: set[str] = set()
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def fetch_page(url: str, user_agent: str, timeout: float) -> str:
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    response.raise_for_status()
    return response.text


def extension_for(content_type: str) -> str:
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    return ".jpg"


def write_metadata_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in METADATA_COLUMNS})


def existing_download_path(output_folder: Path, filename_stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        path = output_folder / f"{filename_stem}{suffix}"
        if path.exists():
            return path
    return None


def download_image(
    image_url: str,
    output_folder: Path,
    filename_stem: str,
    user_agent: str,
    timeout: float,
    min_side: int,
) -> tuple[Path | None, dict[str, Any]]:
    response = requests.get(image_url, headers={"User-Agent": user_agent}, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "image" not in content_type:
        return None, {"status": "skipped", "message": f"non_image_content_type:{content_type}"}

    raw_bytes = response.content
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return None, {"status": "skipped", "message": "unreadable_image"}

    width, height = image.size
    if min(width, height) < min_side:
        return None, {
            "status": "skipped",
            "message": f"too_small:{width}x{height}",
            "width": width,
            "height": height,
            "sha256": sha256,
        }

    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{filename_stem}{extension_for(content_type)}"
    if output_path.suffix.lower() == ".png":
        image.save(output_path, format="PNG")
    elif output_path.suffix.lower() == ".webp":
        image.save(output_path, format="WEBP", quality=92)
    else:
        output_path = output_path.with_suffix(".jpg")
        image.save(output_path, format="JPEG", quality=92)

    return output_path, {
        "status": "downloaded",
        "message": "ok",
        "width": width,
        "height": height,
        "sha256": sha256,
    }


def main() -> None:
    args = parse_args()
    ensure_class_dirs(args.output_root)
    sources = load_sources(args.sources)
    existing_urls = read_existing_metadata(args.metadata)
    class_counts = count_existing_images(args.output_root)
    robots_cache: dict[str, tuple[bool, robotparser.RobotFileParser | None]] = {}
    rows: list[dict[str, Any]] = []
    candidate_counts: dict[str, int] = defaultdict(int)

    for _, source in sources.iterrows():
        class_name = normalize_class_name(source["fashion_mnist_class"])
        source_url = str(source["url"]).strip()
        source_name = str(source["source_name"]).strip() or urlparse(source_url).netloc
        if class_counts[class_name] >= args.target_per_class:
            continue

        allowed, robots_message = can_fetch(
            source_url,
            args.user_agent,
            robots_cache,
            args.allow_missing_robots,
            args.timeout,
        )
        if not allowed:
            rows.append(
                {
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "source_name": source_name,
                    "source_url": source_url,
                    "fashion_mnist_class": class_name,
                    "status": "skipped",
                    "message": robots_message,
                }
            )
            continue

        try:
            html = fetch_page(source_url, args.user_agent, args.timeout)
        except requests.RequestException as exc:
            rows.append(
                {
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "source_name": source_name,
                    "source_url": source_url,
                    "fashion_mnist_class": class_name,
                    "status": "skipped",
                    "message": f"page_fetch_failed:{exc.__class__.__name__}",
                }
            )
            continue
        time.sleep(args.delay)

        image_urls = collect_image_candidates(html, source_url)
        image_urls.extend(collect_shopify_image_candidates(source_url, args.user_agent, args.timeout))
        image_urls = list(dict.fromkeys(image_urls))
        candidate_counts[source_url] = len(image_urls)
        if args.dry_run:
            continue

        per_source_downloaded = 0
        for image_url in image_urls:
            if class_counts[class_name] >= args.target_per_class:
                break
            if per_source_downloaded >= args.max_images_per_source:
                break
            if image_url in existing_urls:
                continue

            digest = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:12]
            filename_stem = f"{slugify(source_name)}_{digest}"
            output_folder = class_dir(args.output_root, class_name)
            existing_path = existing_download_path(output_folder, filename_stem)
            if existing_path:
                existing_urls.add(image_url)
                rows.append(
                    {
                        "downloaded_at": datetime.now(timezone.utc).isoformat(),
                        "source_name": source_name,
                        "source_url": source_url,
                        "fashion_mnist_class": class_name,
                        "image_url": image_url,
                        "filename": str(existing_path),
                        "status": "skipped",
                        "message": "existing_file",
                    }
                )
                continue

            image_allowed, image_robots_message = can_fetch(
                image_url,
                args.user_agent,
                robots_cache,
                args.allow_missing_robots,
                args.timeout,
            )
            if not image_allowed:
                rows.append(
                    {
                        "downloaded_at": datetime.now(timezone.utc).isoformat(),
                        "source_name": source_name,
                        "source_url": source_url,
                        "fashion_mnist_class": class_name,
                        "image_url": image_url,
                        "status": "skipped",
                        "message": image_robots_message,
                    }
                )
                continue

            try:
                output_path, image_info = download_image(
                    image_url,
                    output_folder,
                    filename_stem,
                    args.user_agent,
                    args.timeout,
                    args.min_side,
                )
            except requests.RequestException as exc:
                output_path = None
                image_info = {
                    "status": "skipped",
                    "message": f"image_fetch_failed:{exc.__class__.__name__}",
                }

            rows.append(
                {
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "source_name": source_name,
                    "source_url": source_url,
                    "fashion_mnist_class": class_name,
                    "image_url": image_url,
                    "filename": str(output_path) if output_path else "",
                    **image_info,
                }
            )
            if image_info["status"] == "downloaded":
                class_counts[class_name] += 1
                per_source_downloaded += 1
                existing_urls.add(image_url)
            time.sleep(args.delay)

        write_metadata_rows(args.metadata, rows)
        rows = []

    if args.dry_run:
        for url, count in candidate_counts.items():
            print(f"{count:4d} image candidates: {url}")
        return

    write_metadata_rows(args.metadata, rows)
    print("Download summary:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {class_counts[class_name]} images")
    print(f"Metadata written to {args.metadata}")


if __name__ == "__main__":
    main()
