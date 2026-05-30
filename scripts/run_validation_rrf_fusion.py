from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.utils import read_jsonl, write_json, write_jsonl


def parse_float_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Weight grid must contain at least one value.")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise argparse.ArgumentTypeError("Fusion weights must be in [0, 1].")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-select a weighted RRF fusion between two reranker outputs."
    )
    parser.add_argument("--qrels", default="data/processed/bioasq_qrels.jsonl")
    parser.add_argument("--a-validation", required=True)
    parser.add_argument("--b-validation", required=True)
    parser.add_argument("--a-test", required=True)
    parser.add_argument("--b-test", required=True)
    parser.add_argument("--a-label", default="source_a")
    parser.add_argument("--b-label", default="source_b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--table-output", required=True)
    parser.add_argument("--weight-grid", type=parse_float_grid, default=parse_float_grid("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"))
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--primary", choices=["mrr@10", "recall@10", "ndcg@10", "composite"], default="composite")
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


def weighted_rrf_fusion(
    a_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
    *,
    a_weight: float,
    rrf_k: int,
    top_k: int,
    retriever_name: str,
) -> list[dict[str, Any]]:
    a_by_qid = group_predictions(a_rows)
    b_by_qid = group_predictions(b_rows)
    qids = sorted(set(a_by_qid) | set(b_by_qid))
    output: list[dict[str, Any]] = []
    b_weight = 1.0 - a_weight
    for qid in qids:
        scores: dict[str, float] = defaultdict(float)
        rank_sources: dict[str, dict[str, int]] = defaultdict(dict)
        row_lookup: dict[str, dict[str, Any]] = {}
        for source_name, rows, weight in [
            ("a", a_by_qid.get(qid, []), a_weight),
            ("b", b_by_qid.get(qid, []), b_weight),
        ]:
            for row in rows:
                rank = int(row.get("rank", top_k + 1))
                if rank > top_k:
                    continue
                pid = str(row["passage_id"])
                scores[pid] += weight / (rrf_k + rank)
                rank_sources[pid][source_name] = rank
                row_lookup.setdefault(pid, row)
        ranked = sorted(
            scores,
            key=lambda pid: (
                -scores[pid],
                min(rank_sources[pid].values()),
                pid,
            ),
        )
        for rank, pid in enumerate(ranked[:top_k], start=1):
            base_row = row_lookup[pid]
            metadata = dict(base_row.get("metadata", {}))
            metadata["validation_rrf_fusion"] = {
                "a_weight": a_weight,
                "b_weight": b_weight,
                "rrf_k": rrf_k,
                "a_rank": rank_sources[pid].get("a"),
                "b_rank": rank_sources[pid].get("b"),
            }
            output.append(
                {
                    "question_id": qid,
                    "passage_id": pid,
                    "rank": rank,
                    "score": float(scores[pid]),
                    "retriever": retriever_name,
                    "metadata": metadata,
                }
            )
    return output


def selection_score(metrics: dict[str, Any], primary: str) -> tuple[float, float, float, float]:
    recall = float(metrics.get("recall@10", 0.0))
    ndcg = float(metrics.get("ndcg@10", 0.0))
    mrr = float(metrics.get("mrr@10", 0.0))
    if primary == "composite":
        return (recall + 0.5 * ndcg + 0.2 * mrr, recall, ndcg, mrr)
    return (float(metrics.get(primary, 0.0)), recall, ndcg, mrr)


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
    a_validation = read_jsonl(args.a_validation)
    b_validation = read_jsonl(args.b_validation)
    a_test = read_jsonl(args.a_test)
    b_test = read_jsonl(args.b_test)

    validation_qids = set(group_predictions(a_validation)) & set(group_predictions(b_validation))
    test_qids = set(group_predictions(a_test)) & set(group_predictions(b_test))
    validation_qrels = filter_qrels(qrels, validation_qids)
    test_qrels = filter_qrels(qrels, test_qids)

    trials = []
    for weight in args.weight_grid:
        predictions = weighted_rrf_fusion(
            a_validation,
            b_validation,
            a_weight=weight,
            rrf_k=args.rrf_k,
            top_k=args.top_k,
            retriever_name="validation_rrf_fusion_validation",
        )
        metrics = evaluate_retrieval(validation_qrels, predictions, sorted(set(args.ks)))
        trials.append({"a_weight": weight, "metrics": metrics, "selection_score": selection_score(metrics, args.primary)})

    trials.sort(key=lambda row: tuple(-value for value in row["selection_score"]))
    selected_weight = float(trials[0]["a_weight"])
    test_predictions = weighted_rrf_fusion(
        a_test,
        b_test,
        a_weight=selected_weight,
        rrf_k=args.rrf_k,
        top_k=args.top_k,
        retriever_name="validation_rrf_fusion",
    )
    write_jsonl(args.output, test_predictions)

    a_metrics = evaluate_retrieval(test_qrels, a_test, sorted(set(args.ks)))
    b_metrics = evaluate_retrieval(test_qrels, b_test, sorted(set(args.ks)))
    fused_metrics = evaluate_retrieval(test_qrels, test_predictions, sorted(set(args.ks)))
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "qrels": args.qrels,
        "a_label": args.a_label,
        "b_label": args.b_label,
        "a_validation": args.a_validation,
        "b_validation": args.b_validation,
        "a_test": args.a_test,
        "b_test": args.b_test,
        "primary": args.primary,
        "rrf_k": args.rrf_k,
        "top_k": args.top_k,
        "selected_a_weight": selected_weight,
        "validation_trials": trials,
        "a_test_metrics": a_metrics,
        "b_test_metrics": b_metrics,
        "fused_test_metrics": fused_metrics,
        "output": args.output,
    }
    write_json(args.metrics_output, payload)
    write_table(
        args.table_output,
        [
            metric_row(args.a_label, a_metrics),
            metric_row(args.b_label, b_metrics),
            metric_row(f"Validation RRF fusion a={selected_weight:.2f}", fused_metrics),
        ],
    )
    print(
        {
            "selected_a_weight": selected_weight,
            "metrics": args.metrics_output,
            "output": args.output,
            "fusion_recall@10": fused_metrics.get("recall@10"),
            "fusion_mrr@10": fused_metrics.get("mrr@10"),
            "fusion_ndcg@10": fused_metrics.get("ndcg@10"),
        }
    )


if __name__ == "__main__":
    main()
