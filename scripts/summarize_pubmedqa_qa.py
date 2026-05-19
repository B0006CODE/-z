from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PubMedQA answer-selection metric JSON files.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "results/metrics/pubmedqa_dense_qa_test_metrics.json",
            "results/metrics/pubmedqa_hybrid_qa_test_metrics.json",
            "results/metrics/pubmedqa_hgb_qa_test_metrics.json",
        ],
    )
    parser.add_argument("--output", default="results/tables/pubmedqa_qa_accuracy.md")
    return parser.parse_args()


def load_results(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = []
    for result in payload.get("results", []):
        row = dict(result)
        row["source_metrics"] = path
        rows.append(row)
    return rows


def method_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    method_order = {"majority": 0, "lexical_rule": 1, "tfidf_logreg": 2}
    return (
        str(row.get("prediction_name", "")),
        int(row.get("top_k", 0)),
        f"{method_order.get(str(row.get('method')), 99):02d}_{row.get('method')}",
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| evidence_source | method | top_k | num_eval | evidence_hit@k | accuracy | macro_f1 | yes_f1 | no_f1 | maybe_f1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=method_sort_key):
        metrics = row["metrics"]
        per_label = metrics["per_label"]
        lines.append(
            "| {source} | {method} | {top_k} | {num_eval} | {coverage:.4f} | {accuracy:.4f} | {macro_f1:.4f} | {yes_f1:.4f} | {no_f1:.4f} | {maybe_f1:.4f} |".format(
                source=row["prediction_name"],
                method=row["method"],
                top_k=row["top_k"],
                num_eval=row["num_eval"],
                coverage=row["evidence_hit_at_k"],
                accuracy=metrics["accuracy"],
                macro_f1=metrics["macro_f1"],
                yes_f1=per_label["yes"]["f1"],
                no_f1=per_label["no"]["f1"],
                maybe_f1=per_label["maybe"]["f1"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.metrics:
        rows.extend(load_results(path))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown_table(rows), encoding="utf-8")
    print({"output": str(output), "num_rows": len(rows)})


if __name__ == "__main__":
    main()
