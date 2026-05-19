from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_KS = [1, 3, 5, 10]
DEFAULT_METRICS = ["recall", "mrr", "ndcg"]


def parse_dataset_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Dataset input must use Dataset=metrics.json format.")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("Dataset label and path must be non-empty.")
    return label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize held-out top-k sensitivity for HGB vs Hybrid.")
    parser.add_argument("--dataset", action="append", type=parse_dataset_arg, required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--baseline-label", default="Hybrid RRF")
    parser.add_argument("--candidate-label", default="HGB all features")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def format_float(value: Any, signed: bool = False) -> str:
    if not isinstance(value, (float, int)):
        return ""
    if signed:
        return f"{float(value):+.4f}"
    return f"{float(value):.4f}"


def metric_label(metric: str, k: int) -> str:
    if metric == "mrr":
        return f"MRR@{k}"
    if metric == "ndcg":
        return f"nDCG@{k}"
    return f"{metric.capitalize()}@{k}"


def build_rows(
    datasets: list[tuple[str, str]],
    ks: list[int],
    metrics: list[str],
    baseline_label: str,
    candidate_label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_label, path in datasets:
        candidate = read_json(path)
        baseline = candidate.get("source_test_metrics", {})
        if not baseline:
            raise ValueError(f"Metrics file lacks source_test_metrics: {path}")
        for k in ks:
            row: dict[str, Any] = {
                "dataset": dataset_label,
                "k": k,
                "baseline": baseline_label,
                "candidate": candidate_label,
                "num_questions": candidate.get("num_questions_with_qrels"),
            }
            for metric in metrics:
                key = f"{metric}@{k}"
                baseline_value = baseline.get(key)
                candidate_value = candidate.get(key)
                delta = (
                    float(candidate_value) - float(baseline_value)
                    if isinstance(baseline_value, (float, int)) and isinstance(candidate_value, (float, int))
                    else None
                )
                row[f"baseline_{metric}"] = baseline_value
                row[f"candidate_{metric}"] = candidate_value
                row[f"delta_{metric}"] = delta
            rows.append(row)
    return rows


def write_markdown(path: str | Path, rows: list[dict[str, Any]], metrics: list[str]) -> None:
    headers = ["Dataset", "k", "n"]
    for metric in metrics:
        label = metric_label(metric, 0).replace("@0", "")
        headers.extend([f"Hybrid {label}", f"HGB {label}", f"Delta {label}"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [str(row["dataset"]), str(row["k"]), str(row.get("num_questions", ""))]
        for metric in metrics:
            cells.extend(
                [
                    format_float(row.get(f"baseline_{metric}")),
                    format_float(row.get(f"candidate_{metric}")),
                    format_float(row.get(f"delta_{metric}"), signed=True),
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Rows use held-out test splits and compare the supervised HGB reranker with the matched Hybrid RRF source metrics."
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(
        datasets=args.dataset,
        ks=sorted(set(args.ks)),
        metrics=args.metrics,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows, args.metrics)
    print({"output_csv": args.output_csv, "output_md": args.output_md, "num_rows": len(rows)})


if __name__ == "__main__":
    main()
