from __future__ import annotations

import argparse
import os
from pathlib import Path

mpl_dir = Path.cwd() / ".matplotlib"
mpl_dir.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from torchvision.datasets import FashionMNIST

from retail_common import CLASS_NAMES, METADATA_DIR, RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create EDA figures for the retailer image dataset.")
    parser.add_argument("--validated", type=Path, default=METADATA_DIR / "validated_images.csv")
    parser.add_argument("--splits", type=Path, default=METADATA_DIR / "splits.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--examples-per-class", type=int, default=3)
    return parser.parse_args()


def load_usable(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Validated metadata not found: {path}")
    frame = pd.read_csv(path)
    if "usable" not in frame.columns:
        raise ValueError("Validated metadata must include a 'usable' column.")
    usable = frame[frame["usable"]].copy()
    if usable.empty:
        raise ValueError("No usable retailer images found.")
    return usable


def plot_class_balance(frame: pd.DataFrame, output_dir: Path) -> None:
    counts = (
        frame["fashion_mnist_class"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
        .rename_axis("class")
        .reset_index(name="count")
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=counts, x="class", y="count", color="#2A9D8F", ax=ax)
    ax.set_title("Usable retailer images per Fashion-MNIST class")
    ax.set_xlabel("")
    ax.set_ylabel("Image count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "retail_class_balance.png", dpi=180)
    plt.close(fig)


def plot_image_sizes(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=frame,
        x="width",
        y="height",
        hue="fashion_mnist_class",
        s=35,
        alpha=0.75,
        ax=ax,
    )
    ax.set_title("Retailer image dimensions")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "retail_image_sizes.png", dpi=180)
    plt.close(fig)


def plot_retail_examples(frame: pd.DataFrame, output_dir: Path, examples_per_class: int) -> None:
    examples = []
    for class_name in CLASS_NAMES:
        rows = frame[frame["fashion_mnist_class"] == class_name].head(examples_per_class)
        for _, row in rows.iterrows():
            examples.append((class_name, Path(row["path"])))

    if not examples:
        return

    fig, axes = plt.subplots(
        len(CLASS_NAMES),
        examples_per_class,
        figsize=(examples_per_class * 3.0, len(CLASS_NAMES) * 2.2),
    )
    if examples_per_class == 1:
        axes = axes[:, None]

    for row_idx, class_name in enumerate(CLASS_NAMES):
        class_examples = [path for label, path in examples if label == class_name]
        for col_idx in range(examples_per_class):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx < len(class_examples):
                with Image.open(class_examples[col_idx]) as image:
                    ax.imshow(image.convert("RGB"))
            if col_idx == 0:
                ax.set_ylabel(class_name, rotation=0, labelpad=45, va="center")

    fig.suptitle("Retailer image examples by mapped Fashion-MNIST class", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "retail_examples_grid.png", dpi=180)
    plt.close(fig)


def plot_fashion_vs_retail(frame: pd.DataFrame, output_dir: Path) -> None:
    fashion = FashionMNIST(root=Path("data"), train=False, download=True)
    fig, axes = plt.subplots(len(CLASS_NAMES), 2, figsize=(7, len(CLASS_NAMES) * 2.0))

    for class_idx, class_name in enumerate(CLASS_NAMES):
        fashion_index = int((fashion.targets == class_idx).nonzero()[0][0])
        axes[class_idx, 0].imshow(fashion.data[fashion_index], cmap="gray")
        axes[class_idx, 0].axis("off")
        axes[class_idx, 0].set_ylabel(class_name, rotation=0, labelpad=45, va="center")
        if class_idx == 0:
            axes[class_idx, 0].set_title("Fashion-MNIST")

        retail_rows = frame[frame["fashion_mnist_class"] == class_name]
        axes[class_idx, 1].axis("off")
        if not retail_rows.empty:
            with Image.open(retail_rows.iloc[0]["path"]) as image:
                axes[class_idx, 1].imshow(image.convert("RGB"))
        if class_idx == 0:
            axes[class_idx, 1].set_title("Retailer image")

    fig.suptitle("Domain gap between benchmark and retailer images", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "fashion_vs_retail_examples.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_usable(args.validated)
    plot_class_balance(frame, args.output_dir)
    plot_image_sizes(frame, args.output_dir)
    plot_retail_examples(frame, args.output_dir, args.examples_per_class)
    plot_fashion_vs_retail(frame, args.output_dir)
    print(f"EDA figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
