from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval, group_predictions, group_qrels
from src.utils import read_jsonl, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-selected score fusion for retrieval/rerank outputs.")
    parser.add_argument("--qrels", default="data/processed/bioasq_qrels.jsonl")
    parser.add_argument("--validation-sources", nargs="+", required=True, help="name=path entries.")
    parser.add_argument("--test-sources", nargs="+", required=True, help="name=path entries matching validation names.")
    parser.add_argument("--output-prefix", default="kch_medrank_fusion")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--selection-primary", choices=["recall", "mrr", "ndcg"], default="recall")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_sources(entries: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Source must be name=path: {entry}")
        name, path = entry.split("=", 1)
        if not name:
            raise ValueError(f"Empty source name in {entry}")
        parsed[name] = path
    return parsed


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def source_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    rows = sorted(rows, key=lambda row: int(row.get("rank", 10**9)))
    raw_scores = [float(row.get("score", 0.0)) for row in rows]
    norm_scores = minmax(raw_scores)
    scores: dict[str, float] = {}
    for row, norm_score in zip(rows, norm_scores, strict=False):
        pid = str(row["passage_id"])
        rank_score = 1.0 / (60.0 + float(row.get("rank", 10**9)))
        scores[pid] = 0.5 * norm_score + 0.5 * rank_score
    return scores


def weight_grid(num_sources: int, step: float) -> list[tuple[float, ...]]:
    if step <= 0 or step > 1:
        raise ValueError("--weight-step must be in (0, 1].")
    units = int(round(1.0 / step))
    if abs(units * step - 1.0) > 1e-9:
        raise ValueError("--weight-step must divide 1.0 exactly, e.g. 0.1 or 0.05.")
    weights: list[tuple[float, ...]] = []
    for combo in itertools.product(range(units + 1), repeat=num_sources):
        if sum(combo) == units:
            weights.append(tuple(value / units for value in combo))
    return weights


def fuse_predictions(
    source_rows: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    *,
    top_k: int,
    retriever_name: str,
) -> list[dict[str, Any]]:
    grouped = {name: group_predictions(rows) for name, rows in source_rows.items()}
    qids = sorted(set.intersection(*(set(rows_by_qid) for rows_by_qid in grouped.values())))
    fused: list[dict[str, Any]] = []
    for qid in qids:
        per_source = {name: source_scores(rows_by_qid[qid]) for name, rows_by_qid in grouped.items()}
        pids = sorted(set.union(*(set(scores) for scores in per_source.values())))
        scored = []
        for pid in pids:
            score = sum(weights[name] * per_source[name].get(pid, 0.0) for name in weights)
            best_rank = min(
                (
                    int(row.get("rank", 10**9))
                    for name, rows_by_qid in grouped.items()
                    for row in rows_by_qid[qid]
                    if str(row["passage_id"]) == pid
                ),
                default=10**9,
            )
            scored.append((score, best_rank, pid))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        for rank, (score, _best_rank, pid) in enumerate(scored[:top_k], start=1):
            fused.append(
                {
                    "question_id": qid,
                    "passage_id": pid,
                    "rank": rank,
                    "score": float(score),
                    "retriever": retriever_name,
                    "metadata": {"weights": weights},
                }
            )
    return fused


def selection_key(metrics: dict[str, Any], primary: str) -> tuple[float, float, float]:
    recall = float(metrics.get("recall@10", 0.0))
    mrr = float(metrics.get("mrr@10", 0.0))
    ndcg = float(metrics.get("ndcg@10", 0.0))
    if primary == "mrr":
        return (mrr, recall, ndcg)
    if primary == "ndcg":
        return (ndcg, mrr, recall)
    return (recall, mrr, ndcg)


def markdown_summary(payload: dict[str, Any]) -> str:
    rows = payload["trials"][:10]
    headers = ["rank", "weights", "val_recall@10", "val_mrr@10", "val_ndcg@10"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for idx, row in enumerate(rows, start=1):
        weights = ", ".join(f"{name}:{value:.2f}" for name, value in row["weights"].items())
        lines.append(
            f"| {idx} | {weights} | {row['metrics'].get('recall@10', 0.0):.4f} | "
            f"{row['metrics'].get('mrr@10', 0.0):.4f} | {row['metrics'].get('ndcg@10', 0.0):.4f} |"
        )
    lines.append("")
    selected = payload["selected"]
    test = payload["test_metrics"]
    selected_weights = ", ".join(f"{name}:{value:.2f}" for name, value in selected["weights"].items())
    lines.append(f"Selected validation weights: {selected_weights}.")
    lines.append(
        f"Test Recall@10={test.get('recall@10', 0.0):.4f}, "
        f"MRR@10={test.get('mrr@10', 0.0):.4f}, nDCG@10={test.get('ndcg@10', 0.0):.4f}."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    qrels = read_jsonl(args.qrels)
    qrels_by_qid = group_qrels(qrels)
    validation_paths = parse_sources(args.validation_sources)
    test_paths = parse_sources(args.test_sources)
    if set(validation_paths) != set(test_paths):
        raise ValueError("Validation and test source names must match.")

    validation_rows = {name: read_jsonl(path) for name, path in validation_paths.items()}
    test_rows = {name: read_jsonl(path) for name, path in test_paths.items()}
    source_names = list(validation_paths)
    validation_qids = set.intersection(*(set(group_predictions(rows)) for rows in validation_rows.values()))
    test_qids = set.intersection(*(set(group_predictions(rows)) for rows in test_rows.values()))
    validation_qrels = [row for row in qrels if str(row["question_id"]) in validation_qids]
    test_qrels = [row for row in qrels if str(row["question_id"]) in test_qids]

    trials: list[dict[str, Any]] = []
    for weight_tuple in weight_grid(len(source_names), args.weight_step):
        weights = dict(zip(source_names, weight_tuple, strict=True))
        predictions = fuse_predictions(
            validation_rows,
            weights,
            top_k=args.top_k,
            retriever_name=f"{args.output_prefix}_validation",
        )
        metrics = evaluate_retrieval(validation_qrels, predictions, sorted(set(args.ks)))
        trials.append({"weights": weights, "metrics": metrics})
    trials.sort(key=lambda row: selection_key(row["metrics"], args.selection_primary), reverse=True)
    selected = trials[0]

    output_dir = Path("outputs/rerank")
    metrics_dir = Path("results/metrics")
    tables_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    validation_predictions = fuse_predictions(
        validation_rows,
        selected["weights"],
        top_k=args.top_k,
        retriever_name=f"{args.output_prefix}_validation_selected",
    )
    test_predictions = fuse_predictions(
        test_rows,
        selected["weights"],
        top_k=args.top_k,
        retriever_name=f"{args.output_prefix}_validation_selected",
    )
    validation_metrics = evaluate_retrieval(validation_qrels, validation_predictions, sorted(set(args.ks)))
    test_metrics = evaluate_retrieval(test_qrels, test_predictions, sorted(set(args.ks)))

    validation_output = output_dir / f"{args.output_prefix}_validation_top{args.top_k}.jsonl"
    test_output = output_dir / f"{args.output_prefix}_test_top{args.top_k}.jsonl"
    write_jsonl(validation_output, validation_predictions)
    write_jsonl(test_output, test_predictions)

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "qrels": args.qrels,
        "validation_sources": validation_paths,
        "test_sources": test_paths,
        "source_names": source_names,
        "selection_primary": args.selection_primary,
        "weight_step": args.weight_step,
        "num_validation_qids": len(validation_qids),
        "num_test_qids": len(test_qids),
        "selected": selected,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_predictions": str(validation_output),
        "test_predictions": str(test_output),
        "trials": trials,
    }
    metrics_path = metrics_dir / f"{args.output_prefix}_metrics.json"
    table_path = tables_dir / f"{args.output_prefix}_validation_selection.md"
    write_json(metrics_path, payload)
    table_path.write_text(markdown_summary(payload), encoding="utf-8")
    print(
        {
            "metrics": str(metrics_path),
            "table": str(table_path),
            "validation_predictions": str(validation_output),
            "test_predictions": str(test_output),
            "selected_weights": selected["weights"],
            "test_recall@10": test_metrics.get("recall@10"),
            "test_mrr@10": test_metrics.get("mrr@10"),
            "test_ndcg@10": test_metrics.get("ndcg@10"),
        }
    )


if __name__ == "__main__":
    main()
