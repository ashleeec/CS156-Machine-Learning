from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

mpl_dir = Path.cwd() / ".matplotlib"
mpl_dir.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import FashionMNIST

from retail_common import CLASS_NAMES, METADATA_DIR, RESULTS_DIR, SIMPLIFIED6_CLASS_NAMES


class RetailSplitDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        class_names: list[str],
        label_column: str,
        transform: transforms.Compose | None = None,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.class_names = class_names
        self.label_column = label_column
        self.transform = transform
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[row[self.label_column]]
        return image, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained image model on real-world retailer images."
    )
    parser.add_argument("--splits", type=Path, default=METADATA_DIR / "splits.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--label-column",
        default="fashion_mnist_class",
        help="Column in the split CSV to use as the training target.",
    )
    parser.add_argument(
        "--model",
        choices=["mobilenet_v3_small", "resnet18"],
        default="mobilenet_v3_small",
        help="Pretrained torchvision backbone to fine-tune.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one tiny training pass for pipeline validation.",
    )
    parser.add_argument(
        "--skip-fashionmnist-eval",
        action="store_true",
        help="Skip evaluating the fine-tuned model on Fashion-MNIST adapted to RGB.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def class_names_for_label_column(frame: pd.DataFrame, label_column: str) -> list[str]:
    values = set(frame[label_column].dropna().astype(str))
    if label_column == "fashion_mnist_class":
        return [class_name for class_name in CLASS_NAMES if class_name in values]
    if label_column == "simplified_class":
        return [class_name for class_name in SIMPLIFIED6_CLASS_NAMES if class_name in values]
    return sorted(values)


def load_splits(path: Path, label_column: str, smoke: bool, seed: int) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Split metadata does not exist: {path}. Run scripts/validate_retail_dataset.py first."
        )
    frame = pd.read_csv(path)
    required = {"path", label_column, "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Split metadata is missing columns: {sorted(missing)}")
    frame = frame[frame["split"].isin(["train", "val", "test"])].copy()
    frame[label_column] = frame[label_column].astype(str)
    class_names = class_names_for_label_column(frame, label_column)
    frame = frame[frame[label_column].isin(class_names)].copy()
    if frame.empty:
        raise ValueError("Split metadata has no usable train/val/test rows.")
    split_coverage = frame.groupby("split")[label_column].nunique()
    missing_class_splits = [
        split for split in ["train", "val", "test"] if split_coverage.get(split, 0) != len(class_names)
    ]
    if missing_class_splits:
        raise ValueError(
            f"Every split must contain all classes. Missing coverage in: {missing_class_splits}"
        )
    missing_files = [path for path in frame["path"] if not Path(path).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing image files, first example: {missing_files[0]}")
    if smoke:
        pieces = []
        for split in ["train", "val", "test"]:
            split_rows = frame[frame["split"] == split]
            pieces.append(split_rows.groupby(label_column, group_keys=False).head(2))
        frame = pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=seed)
    return frame, class_names


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_loaders(
    frame: pd.DataFrame,
    class_names: list[str],
    label_column: str,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, DataLoader], dict[str, RetailSplitDataset]]:
    train_transform, eval_transform = build_transforms()
    datasets = {
        "train": RetailSplitDataset(
            frame[frame["split"] == "train"], class_names, label_column, train_transform
        ),
        "val": RetailSplitDataset(
            frame[frame["split"] == "val"], class_names, label_column, eval_transform
        ),
        "test": RetailSplitDataset(
            frame[frame["split"] == "test"], class_names, label_column, eval_transform
        ),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, dataset in datasets.items()
    }
    return loaders, datasets


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    if model_name == "mobilenet_v3_small":
        for parameter in model.features.parameters():
            parameter.requires_grad = trainable
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return
    if model_name == "resnet18":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = trainable or name.startswith("fc.")
        return
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate(model: nn.Module, loader: DataLoader, run_device: torch.device) -> dict[str, object]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(run_device)
            targets = targets.to(run_device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_stage(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    run_device: torch.device,
    stage_name: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> tuple[nn.Module, pd.DataFrame]:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_macro_f1 = -np.inf
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        for inputs, targets in loaders["train"]:
            inputs = inputs.to(run_device)
            targets = targets.to(run_device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(loaders["train"].dataset)
        val_metrics = evaluate(model, loaders["val"], run_device)
        row = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": time.time() - start,
        }
        history.append(row)
        print(
            f"{stage_name} epoch {epoch}: "
            f"train_loss={train_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1 + 1e-4:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def plot_history(history: pd.DataFrame, output_dir: Path, model_name: str, label_column: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for stage_name, stage_rows in history.groupby("stage"):
        ax.plot(
            stage_rows.index + 1,
            stage_rows["val_macro_f1"],
            marker="o",
            label=f"{stage_name} validation macro F1",
        )
    ax.set_xlabel("Training checkpoint")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{model_name} retail fine-tuning validation performance ({label_column})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    class_names: list[str],
    model_name: str,
    label_column: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Fine-tuned {model_name} on retailer test images ({label_column})")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def evaluate_fashion_mnist(
    model: nn.Module,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    run_device: torch.device,
    class_names: list[str],
) -> dict[str, float]:
    if class_names != CLASS_NAMES:
        return {}
    _, eval_transform = build_transforms()

    def pil_rgb_transform(image: Image.Image) -> torch.Tensor:
        return eval_transform(image.convert("RGB"))

    dataset = FashionMNIST(
        root=Path("data"),
        train=False,
        download=True,
        transform=pil_rgb_transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    metrics = evaluate(model, loader, run_device)
    summary = {
        "fashion_mnist_accuracy": float(metrics["accuracy"]),
        "fashion_mnist_macro_f1": float(metrics["macro_f1"]),
    }
    (output_dir / "fashion_mnist_cross_domain.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_device = device()
    print(f"Using device: {run_device}")

    frame, class_names = load_splits(args.splits, args.label_column, args.smoke, args.seed)
    loaders, _ = build_loaders(
        frame, class_names, args.label_column, args.batch_size, args.num_workers
    )
    model = build_model(args.model, len(class_names)).to(run_device)

    if args.smoke:
        args.head_epochs = 1
        args.finetune_epochs = 0

    set_backbone_trainable(model, args.model, trainable=False)
    model, head_history = train_stage(
        model,
        loaders,
        run_device,
        "frozen_head",
        args.head_epochs,
        args.lr_head,
        args.weight_decay,
        args.patience,
    )

    histories = [head_history]
    if args.finetune_epochs > 0:
        set_backbone_trainable(model, args.model, trainable=True)
        model, finetune_history = train_stage(
            model,
            loaders,
            run_device,
            "finetune_backbone",
            args.finetune_epochs,
            args.lr_finetune,
            args.weight_decay,
            args.patience,
        )
        histories.append(finetune_history)

    history = pd.concat(histories, ignore_index=True)
    history.to_csv(args.output_dir / "training_history.csv", index=False)
    test_metrics = evaluate(model, loaders["test"], run_device)
    metrics_summary = {
        "retail_test_accuracy": float(test_metrics["accuracy"]),
        "retail_test_macro_f1": float(test_metrics["macro_f1"]),
        "class_names": class_names,
        "label_column": args.label_column,
        "model": args.model,
        "split_path": str(args.splits),
    }
    if not args.skip_fashionmnist_eval and not args.smoke:
        metrics_summary.update(
            evaluate_fashion_mnist(
                model,
                args.output_dir,
                args.batch_size,
                args.num_workers,
                run_device,
                class_names,
            )
        )

    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2),
        encoding="utf-8",
    )
    report = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(args.output_dir / "classification_report.csv")
    torch.save(model.state_dict(), args.output_dir / f"{args.model}_retail_finetuned.pt")
    plot_history(history, args.output_dir, args.model, args.label_column)
    plot_confusion(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        args.output_dir,
        class_names,
        args.model,
        args.label_column,
    )

    print(json.dumps(metrics_summary, indent=2))
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
