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

from src.evaluation.retrieval_metrics import evaluate_retrieval, group_qrels
from src.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-select a query-adaptive gate between flat LTR and hypergraph LTR.")
    parser.add_argument("--qrels", default="data/processed/bioasq_qrels.jsonl")
    parser.add_argument("--flat-validation", required=True)
    parser.add_argument("--hyper-validation", required=True)
    parser.add_argument("--flat-test", required=True)
    parser.add_argument("--hyper-test", required=True)
    parser.add_argument("--output", default="outputs/rerank/query_adaptive_hypergraph_gate_test_top100.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/query_adaptive_hypergraph_gate_metrics.json")
    parser.add_argument("--table-output", default="results/tables/query_adaptive_hypergraph_gate.md")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--primary", choices=["mrr@10", "recall@10", "ndcg@10"], default="mrr@10")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    return parser.parse_args()


def group_predictions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    for items in grouped.values():
        items.sort(key=lambda row: int(row["rank"]))
    return dict(grouped)


def feature(row: dict[str, Any], name: str) -> float:
    return float(row.get("metadata", {}).get("features", {}).get(name, 0.0))


def query_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    top10 = rows[:10]
    top20 = rows[:20]
    return {
        "max_hypergraph": max((feature(row, "hypergraph_score_norm") for row in top20), default=0.0),
        "mean_hypergraph": sum(feature(row, "hypergraph_score_norm") for row in top20) / len(top20) if top20 else 0.0,
        "max_mesh_overlap": max((feature(row, "mesh_overlap_count") for row in top10), default=0.0),
        "max_entity_overlap": max((feature(row, "entity_overlap_count") for row in top10), default=0.0),
        "max_mesh_hierarchy": max((feature(row, "question_mesh_hierarchy_coverage") for row in top20), default=0.0),
        "max_shared_cluster": max((feature(row, "shared_mesh_term_cluster_size") for row in top20), default=0.0),
        "mean_base_rank": sum(float(row.get("metadata", {}).get("base_rank") or row["rank"]) for row in top10) / len(top10) if top10 else 0.0,
    }


def use_hypergraph(stats: dict[str, float], rule: dict[str, float]) -> bool:
    low_direct = (stats["max_mesh_overlap"] + stats["max_entity_overlap"]) <= rule["direct_overlap_max"]
    structural_signal = (
        stats["max_hypergraph"] >= rule["hypergraph_min"]
        or stats["max_mesh_hierarchy"] >= rule["mesh_hierarchy_min"]
        or stats["max_shared_cluster"] >= rule["shared_cluster_min"]
    )
    rank_disagreement = stats["mean_base_rank"] >= rule["mean_base_rank_min"]
    if rule["mode"] == 0:
        return structural_signal
    if rule["mode"] == 1:
        return low_direct and structural_signal
    if rule["mode"] == 2:
        return (low_direct or rank_disagreement) and structural_signal
    return low_direct or structural_signal or rank_disagreement


def gated_predictions(
    flat_rows: list[dict[str, Any]],
    hyper_rows: list[dict[str, Any]],
    rule: dict[str, float],
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flat_by_qid = group_predictions(flat_rows)
    hyper_by_qid = group_predictions(hyper_rows)
    output: list[dict[str, Any]] = []
    gate_counts = {"flat": 0, "hypergraph": 0}
    for qid in sorted(set(flat_by_qid) | set(hyper_by_qid)):
        hyper_items = hyper_by_qid.get(qid, [])
        stats = query_stats(hyper_items)
        selected_hyper = bool(hyper_items) and use_hypergraph(stats, rule)
        source = hyper_items if selected_hyper else flat_by_qid.get(qid, [])
        gate_counts["hypergraph" if selected_hyper else "flat"] += 1
        for rank, row in enumerate(source[:top_k], start=1):
            output.append(
                {
                    **row,
                    "rank": rank,
                    "retriever": "query_adaptive_hypergraph_gate",
                    "metadata": {
                        **row.get("metadata", {}),
                        "gate_selected": "hypergraph" if selected_hyper else "flat",
                        "gate_rule": rule,
                        "gate_query_stats": stats,
                    },
                }
            )
    return output, gate_counts


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def candidate_rules() -> list[dict[str, float]]:
    rules = []
    for mode in [0, 1, 2, 3]:
        for direct_overlap_max in [0.0, 1.0, 2.0]:
            for hypergraph_min in [0.10, 0.20, 0.35, 0.50]:
                for mesh_hierarchy_min in [0.0, 0.25, 0.50]:
                    for shared_cluster_min in [2.0, 4.0, 8.0]:
                        rules.append(
                            {
                                "mode": float(mode),
                                "direct_overlap_max": direct_overlap_max,
                                "hypergraph_min": hypergraph_min,
                                "mesh_hierarchy_min": mesh_hierarchy_min,
                                "shared_cluster_min": shared_cluster_min,
                                "mean_base_rank_min": 20.0,
                            }
                        )
    return rules


def metric_row(method: str, metrics: dict[str, Any]) -> dict[str, str]:
    row = {"method": method}
    for key in ["recall@5", "recall@10", "mrr@10", "ndcg@10", "recall@100", "mrr@100", "ndcg@100"]:
        row[key] = f"{float(metrics.get(key, 0.0)):.4f}" if key in metrics else ""
    return row


def write_table(path: str | Path, rows: list[dict[str, str]]) -> None:
    columns = ["method", "recall@5", "recall@10", "mrr@10", "ndcg@10", "recall@100", "mrr@100", "ndcg@100"]
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
    flat_validation = read_jsonl(args.flat_validation)
    hyper_validation = read_jsonl(args.hyper_validation)
    flat_test = read_jsonl(args.flat_test)
    hyper_test = read_jsonl(args.hyper_test)
    validation_qids = set(group_predictions(flat_validation)) | set(group_predictions(hyper_validation))
    test_qids = set(group_predictions(flat_test)) | set(group_predictions(hyper_test))
    validation_qrels = filter_qrels(qrels, validation_qids)
    test_qrels = filter_qrels(qrels, test_qids)

    trials = []
    for rule in candidate_rules():
        preds, counts = gated_predictions(flat_validation, hyper_validation, rule, top_k=args.top_k)
        metrics = evaluate_retrieval(validation_qrels, preds, args.ks)
        trials.append({"rule": rule, "metrics": metrics, "gate_counts": counts})
    trials.sort(
        key=lambda row: (
            -float(row["metrics"].get(args.primary, 0.0)),
            -float(row["metrics"].get("recall@10", 0.0)),
            -float(row["metrics"].get("ndcg@10", 0.0)),
        )
    )
    selected = trials[0]
    test_preds, test_gate_counts = gated_predictions(flat_test, hyper_test, selected["rule"], top_k=args.top_k)
    write_jsonl(args.output, test_preds)
    flat_metrics = evaluate_retrieval(test_qrels, flat_test, args.ks)
    hyper_metrics = evaluate_retrieval(test_qrels, hyper_test, args.ks)
    gated_metrics = evaluate_retrieval(test_qrels, test_preds, args.ks)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "primary": args.primary,
        "selected_rule": selected["rule"],
        "selected_validation_metrics": selected["metrics"],
        "selected_validation_gate_counts": selected["gate_counts"],
        "test_gate_counts": test_gate_counts,
        "flat_test_metrics": flat_metrics,
        "hypergraph_test_metrics": hyper_metrics,
        "gated_test_metrics": gated_metrics,
        "top_trials": trials[:10],
        "output": args.output,
    }
    write_json(args.metrics_output, payload)
    write_table(
        args.table_output,
        [
            metric_row("Flat knowledge LTR", flat_metrics),
            metric_row("Full hypergraph LTR", hyper_metrics),
            metric_row("Query-adaptive hypergraph gate", gated_metrics),
        ],
    )
    print(
        {
            "output": args.output,
            "metrics": args.metrics_output,
            "selected_rule": selected["rule"],
            "gated_recall@10": gated_metrics.get("recall@10"),
            "gated_mrr@10": gated_metrics.get("mrr@10"),
            "gate_counts": test_gate_counts,
        }
    )


if __name__ == "__main__":
    main()
