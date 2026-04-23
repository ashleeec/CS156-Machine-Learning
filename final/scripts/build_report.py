from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from retail_common import REPORT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the real-world fine-tuning paper to PDF.")
    parser.add_argument("--template", type=Path, default=REPORT_DIR / "retail_finetuning_paper.tex")
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "retail_finetuning_paper.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.template.suffix != ".tex":
        raise ValueError("Report builds now expect a LaTeX .tex template.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={args.output.parent}",
        str(args.template),
    ]
    subprocess.run(command, check=True)
    subprocess.run(command, check=True)

    rendered_pdf = args.output.parent / f"{args.template.stem}.pdf"
    if rendered_pdf != args.output:
        rendered_pdf.replace(args.output)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
