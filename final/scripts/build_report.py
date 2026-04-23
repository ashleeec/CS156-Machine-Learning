from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

from retail_common import REPORT_DIR, RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the real-world fine-tuning paper to PDF.")
    parser.add_argument("--template", type=Path, default=REPORT_DIR / "retail_finetuning_paper.md")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "retail_finetuning_paper.pdf")
    parser.add_argument("--keep-rendered-md", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_metric(value: object) -> str:
    if value is None or value == "":
        return "TBD"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def figure_block(path: Path, caption: str) -> str:
    if path.exists():
        return f"![{caption}]({path.as_posix()})"
    return f"*Figure pending: {caption}. Run the data collection, EDA, and training scripts to generate it.*"


def table_block(path: Path) -> str:
    if not path.exists():
        return "*Classification report pending. Run the fine-tuning script to generate it.*"
    frame = pd.read_csv(path, index_col=0)
    wanted = [row for row in frame.index if row in {
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
        "macro avg",
        "weighted avg",
    }]
    frame = frame.loc[wanted, [column for column in ["precision", "recall", "f1-score", "support"] if column in frame.columns]]
    return frame.to_markdown(floatfmt=".3f")


def render_markdown(template: Path, results_dir: Path) -> str:
    text = template.read_text(encoding="utf-8")
    metrics = load_json(results_dir / "metrics_summary.json")
    replacements = {
        "{{date}}": date.today().isoformat(),
        "{{retail_test_accuracy}}": format_metric(metrics.get("retail_test_accuracy")),
        "{{retail_test_macro_f1}}": format_metric(metrics.get("retail_test_macro_f1")),
        "{{fashion_mnist_accuracy}}": format_metric(metrics.get("fashion_mnist_accuracy")),
        "{{fashion_mnist_macro_f1}}": format_metric(metrics.get("fashion_mnist_macro_f1")),
        "{{retail_examples_figure}}": figure_block(
            results_dir / "retail_examples_grid.png",
            "Retailer image examples by mapped Fashion-MNIST class",
        ),
        "{{domain_gap_figure}}": figure_block(
            results_dir / "fashion_vs_retail_examples.png",
            "Examples showing the visual domain gap between Fashion-MNIST and retailer images",
        ),
        "{{class_balance_figure}}": figure_block(
            results_dir / "retail_class_balance.png",
            "Usable retailer images per class",
        ),
        "{{training_curve_figure}}": figure_block(
            results_dir / "training_curve.png",
            "Validation macro F1 during transfer learning",
        ),
        "{{confusion_matrix_figure}}": figure_block(
            results_dir / "confusion_matrix.png",
            "Confusion matrix on the retailer-image test set",
        ),
        "{{classification_report_table}}": table_block(results_dir / "classification_report.csv"),
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_markdown(args.template, args.results_dir)
    rendered_path = args.output.with_suffix(".rendered.md")
    rendered_path.write_text(rendered, encoding="utf-8")

    command = [
        "pandoc",
        str(rendered_path),
        "--from",
        "markdown",
        "--pdf-engine=xelatex",
        "-V",
        "geometry:margin=1in",
        "-V",
        "fontsize=11pt",
        "-o",
        str(args.output),
    ]
    subprocess.run(command, check=True)
    if not args.keep_rendered_md:
        rendered_path.unlink()
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
