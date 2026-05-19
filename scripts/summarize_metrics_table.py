from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = ["recall@10", "mrr@10", "ndcg@10", "recall@100", "mrr@100", "ndcg@100"]


def parse_metric_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Metric input must use Label=path.json format.")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("Metric label and path must be non-empty.")
    return label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize selected metrics JSON files into CSV and Markdown tables.")
    parser.add_argument("--metric", action="append", type=parse_metric_arg, required=True, help="Label=metrics.json")
    parser.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def format_value(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return ""


def write_markdown(path: str | Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    header = ["method", *columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in header) + " |")
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for label, path in args.metric:
        metrics = read_json(path)
        row = {"method": label}
        for column in args.columns:
            row[column] = format_value(metrics.get(column))
        rows.append(row)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", *args.columns])
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(args.output_md, rows, args.columns)
    print({"output_csv": args.output_csv, "output_md": args.output_md, "num_rows": len(rows)})


if __name__ == "__main__":
    main()
