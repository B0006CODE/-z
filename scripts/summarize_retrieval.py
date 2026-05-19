from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    ("BM25", "results/metrics/bm25_full_top100_metrics.json"),
    ("Dense", "results/metrics/dense_full_top100_metrics.json"),
    ("Hybrid RRF", "results/metrics/hybrid_full_top100_metrics.json"),
    ("Entity Overlap RRF", "results/metrics/entity_overlap_full_top100_metrics.json"),
]
DEFAULT_COLUMNS = [
    "recall@5",
    "recall@10",
    "recall@20",
    "recall@50",
    "recall@100",
    "mrr@10",
    "mrr@100",
    "ndcg@10",
    "ndcg@100",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize retrieval metrics into CSV and Markdown tables.")
    parser.add_argument("--output-csv", default="results/tables/first_stage_retrieval.csv")
    parser.add_argument("--output-md", default="results/tables/first_stage_retrieval.md")
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def format_float(value: Any) -> str:
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
    for method, path in DEFAULT_METRICS:
        metrics = read_json(path)
        row = {"method": method}
        for column in DEFAULT_COLUMNS:
            row[column] = format_float(metrics.get(column))
        rows.append(row)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", *DEFAULT_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(args.output_md, rows, DEFAULT_COLUMNS)
    print({"output_csv": args.output_csv, "output_md": args.output_md, "num_rows": len(rows)})


if __name__ == "__main__":
    main()
