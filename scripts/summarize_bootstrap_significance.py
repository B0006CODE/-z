from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired bootstrap retrieval tests.")
    parser.add_argument("--input", action="append", required=True, help="Bootstrap JSON file.")
    parser.add_argument("--output", required=True, help="Markdown table output path.")
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def format_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.4f}"


def format_metric(metric: str) -> str:
    return "MRR" if metric == "mrr" else metric.capitalize()


def main() -> None:
    args = parse_args()
    payloads = [read_json(path) for path in args.input]
    headers = [
        "Dataset",
        "Comparison",
        "Metric",
        "k",
        "Baseline",
        "HGB",
        "Delta",
        "95% CI",
        "Relative",
        "p-value",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for payload in payloads:
        comparison = f"{payload['candidate_label']} vs {payload['baseline_label']}"
        for row in payload["results"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        payload["dataset"],
                        comparison,
                        format_metric(row["metric"]),
                        str(row["k"]),
                        f"{row['baseline_mean']:.4f}",
                        f"{row['candidate_mean']:.4f}",
                        f"{row['delta']:+.4f}",
                        f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}]",
                        f"{row['relative_delta_percent']:+.2f}%",
                        format_p(row["p_value_two_sided"]),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("All tests use paired bootstrap over question-level metric values with 10,000 resamples.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"output": str(output), "num_inputs": len(payloads)})


if __name__ == "__main__":
    main()
