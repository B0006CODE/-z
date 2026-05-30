from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-select an anchored fusion between a base ranking and a reranked hypergraph output."
    )
    parser.add_argument("--qrels", default="data/processed/bioasq_qrels.jsonl")
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--candidate-validation", required=True)
    parser.add_argument("--candidate-test", required=True)
    parser.add_argument("--baseline-test", default=None, help="Optional baseline test output for reporting.")
    parser.add_argument("--output", default="outputs/rerank/anchored_hypergraph_fusion_test_top300.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/anchored_hypergraph_fusion_metrics.json")
    parser.add_argument("--table-output", default="results/tables/anchored_hypergraph_fusion.md")
    parser.add_argument("--anchor-grid", type=int, nargs="+", default=[0, 1, 3, 5, 10, 20])
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--primary", choices=["mrr@10", "recall@10", "ndcg@10"], default="ndcg@10")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 50, 100, 200, 300])
    return parser.parse_args()


def group_predictions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    for items in grouped.values():
        items.sort(key=lambda row: int(row.get("rank", 10**9)))
    return dict(grouped)


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def anchored_fusion(
    base_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    anchor_n: int,
    top_k: int,
    retriever_name: str,
) -> list[dict[str, Any]]:
    base_by_qid = group_predictions(base_rows)
    candidate_by_qid = group_predictions(candidate_rows)
    qids = sorted(set(candidate_by_qid))
    output: list[dict[str, Any]] = []
    for qid in qids:
        used: set[str] = set()
        fused: list[dict[str, Any]] = []
        for row in base_by_qid.get(qid, [])[:anchor_n]:
            pid = str(row["passage_id"])
            if pid in used:
                continue
            used.add(pid)
            fused.append(
                {
                    **row,
                    "retriever": retriever_name,
                    "metadata": {
                        **row.get("metadata", {}),
                        "anchor_fusion_source": "base_anchor",
                        "anchor_n": anchor_n,
                    },
                }
            )
        for row in candidate_by_qid.get(qid, []):
            pid = str(row["passage_id"])
            if pid in used:
                continue
            used.add(pid)
            fused.append(
                {
                    **row,
                    "retriever": retriever_name,
                    "metadata": {
                        **row.get("metadata", {}),
                        "anchor_fusion_source": "candidate",
                        "anchor_n": anchor_n,
                    },
                }
            )
            if len(fused) >= top_k:
                break
        if len(fused) < top_k:
            for row in base_by_qid.get(qid, []):
                pid = str(row["passage_id"])
                if pid in used:
                    continue
                used.add(pid)
                fused.append(
                    {
                        **row,
                        "retriever": retriever_name,
                        "metadata": {
                            **row.get("metadata", {}),
                            "anchor_fusion_source": "base_backfill",
                            "anchor_n": anchor_n,
                        },
                    }
                )
                if len(fused) >= top_k:
                    break
        for rank, row in enumerate(fused[:top_k], start=1):
            output.append({**row, "rank": rank})
    return output


def metric_row(method: str, metrics: dict[str, Any]) -> dict[str, str]:
    row = {"method": method}
    for key in ["recall@5", "recall@10", "mrr@10", "ndcg@10", "recall@100", "recall@200", "recall@300"]:
        row[key] = f"{float(metrics.get(key, 0.0)):.4f}" if key in metrics else ""
    return row


def write_table(path: str | Path, rows: list[dict[str, str]]) -> None:
    columns = ["method", "recall@5", "recall@10", "mrr@10", "ndcg@10", "recall@100", "recall@200", "recall@300"]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with target.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    qrels = read_jsonl(args.qrels)
    base_rows = read_jsonl(args.base_predictions)
    candidate_validation = read_jsonl(args.candidate_validation)
    candidate_test = read_jsonl(args.candidate_test)
    baseline_test = read_jsonl(args.baseline_test) if args.baseline_test else []

    validation_qids = set(group_predictions(candidate_validation))
    test_qids = set(group_predictions(candidate_test))
    validation_qrels = filter_qrels(qrels, validation_qids)
    test_qrels = filter_qrels(qrels, test_qids)

    trials = []
    for anchor_n in args.anchor_grid:
        predictions = anchored_fusion(
            base_rows,
            candidate_validation,
            anchor_n=anchor_n,
            top_k=args.top_k,
            retriever_name="anchored_hypergraph_fusion_validation",
        )
        metrics = evaluate_retrieval(validation_qrels, predictions, args.ks)
        trials.append({"anchor_n": anchor_n, "metrics": metrics})

    trials.sort(
        key=lambda row: (
            -float(row["metrics"].get(args.primary, 0.0)),
            -float(row["metrics"].get("recall@10", 0.0)),
            -float(row["metrics"].get("mrr@10", 0.0)),
            int(row["anchor_n"]),
        )
    )
    selected_anchor = int(trials[0]["anchor_n"])
    test_predictions = anchored_fusion(
        base_rows,
        candidate_test,
        anchor_n=selected_anchor,
        top_k=args.top_k,
        retriever_name="anchored_hypergraph_fusion",
    )
    write_jsonl(args.output, test_predictions)

    candidate_metrics = evaluate_retrieval(test_qrels, candidate_test, args.ks)
    fused_metrics = evaluate_retrieval(test_qrels, test_predictions, args.ks)
    baseline_metrics = evaluate_retrieval(test_qrels, baseline_test, args.ks) if baseline_test else {}
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "primary": args.primary,
        "selected_anchor_n": selected_anchor,
        "validation_trials": trials,
        "candidate_test_metrics": candidate_metrics,
        "baseline_test_metrics": baseline_metrics,
        "anchored_fusion_test_metrics": fused_metrics,
        "output": args.output,
    }
    write_json(args.metrics_output, payload)

    rows = []
    if baseline_metrics:
        rows.append(metric_row("Baseline", baseline_metrics))
    rows.append(metric_row("Candidate reranker", candidate_metrics))
    rows.append(metric_row(f"Anchored fusion top{selected_anchor}", fused_metrics))
    write_table(args.table_output, rows)
    print(
        {
            "selected_anchor_n": selected_anchor,
            "metrics": args.metrics_output,
            "output": args.output,
            "fusion_recall@10": fused_metrics.get("recall@10"),
            "fusion_mrr@10": fused_metrics.get("mrr@10"),
            "fusion_ndcg@10": fused_metrics.get("ndcg@10"),
        }
    )


if __name__ == "__main__":
    main()
