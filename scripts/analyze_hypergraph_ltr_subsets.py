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

from src.evaluation.retrieval_metrics import evaluate_retrieval, group_predictions, group_qrels
from src.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rerankers on hard, no-overlap, and multi-evidence query subsets.")
    parser.add_argument("--qrels", default="data/processed/bioasq_qrels.jsonl")
    parser.add_argument("--baseline", required=True, help="Baseline candidate ranking used to define the hard subset.")
    parser.add_argument(
        "--feature-source",
        required=True,
        help="Prediction JSONL with per-candidate features used to define no-overlap structural subsets.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Named prediction path as label=path. May be supplied multiple times.",
    )
    parser.add_argument("--output-json", default="results/metrics/hypergraph_ltr_subset_analysis.json")
    parser.add_argument("--output-md", default="results/tables/hypergraph_ltr_subset_analysis.md")
    parser.add_argument("--baseline-top-k", type=int, default=100)
    parser.add_argument("--miss-top-k", type=int, default=10)
    parser.add_argument("--feature-top-k", type=int, default=200)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 100])
    return parser.parse_args()


def parse_named_paths(raw_items: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Prediction must use label=path format: {item}")
        label, path = item.split("=", 1)
        if not label.strip() or not path.strip():
            raise ValueError(f"Prediction must use label=path format: {item}")
        parsed.append((label.strip(), path.strip()))
    return parsed


def feature_value(row: dict[str, Any], name: str) -> float:
    return float(row.get("metadata", {}).get("features", {}).get(name, 0.0))


def qids_with_predictions(rows: list[dict[str, Any]]) -> set[str]:
    return set(group_predictions(rows))


def hard_subset_qids(qrels: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], *, top_k: int, miss_top_k: int) -> set[str]:
    qrels_by_qid = group_qrels(qrels)
    baseline_by_qid = group_predictions(baseline_rows)
    hard = set()
    for qid, gold in qrels_by_qid.items():
        ranked = [str(row["passage_id"]) for row in baseline_by_qid.get(qid, [])[:top_k]]
        if not ranked:
            continue
        gold_ids = set(gold)
        if gold_ids.intersection(ranked[:miss_top_k]):
            continue
        if gold_ids.intersection(ranked):
            hard.add(qid)
    return hard


def feature_subsets(qrels: list[dict[str, Any]], feature_rows: list[dict[str, Any]], *, top_k: int) -> dict[str, set[str]]:
    qrels_by_qid = group_qrels(qrels)
    feature_by_qid = group_predictions(feature_rows)
    no_direct_mesh = set()
    entity_zero = set()
    high_hypergraph = set()
    shared_cluster = set()
    for qid in qrels_by_qid:
        rows = feature_by_qid.get(qid, [])[:top_k]
        if not rows:
            continue
        if max((feature_value(row, "mesh_overlap_count") for row in rows), default=0.0) <= 0.0:
            no_direct_mesh.add(qid)
        if max((feature_value(row, "entity_overlap_count") for row in rows), default=0.0) <= 0.0:
            entity_zero.add(qid)
        if max((feature_value(row, "hypergraph_score_norm") for row in rows), default=0.0) >= 0.5:
            high_hypergraph.add(qid)
        if max((feature_value(row, "shared_mesh_term_cluster_size") for row in rows), default=0.0) >= 4.0:
            shared_cluster.add(qid)
    multi_evidence = {qid for qid, gold in qrels_by_qid.items() if len(gold) >= 2}
    return {
        "no_direct_mesh_overlap": no_direct_mesh,
        "entity_overlap_zero": entity_zero,
        "high_hypergraph_signal": high_hypergraph,
        "shared_cluster_size_ge4": shared_cluster,
        "multi_evidence": multi_evidence,
    }


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def filter_predictions(predictions: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in predictions if str(row["question_id"]) in qids]


def metric_row(subset_name: str, method: str, subset_size: int, metrics: dict[str, Any]) -> dict[str, str]:
    row = {"subset": subset_name, "n": str(subset_size), "method": method}
    for key in ["recall@5", "recall@10", "mrr@10", "ndcg@10", "recall@100", "mrr@100", "ndcg@100"]:
        row[key] = f"{float(metrics.get(key, 0.0)):.4f}" if key in metrics else ""
    return row


def write_table(path: str | Path, rows: list[dict[str, str]]) -> None:
    columns = ["subset", "n", "method", "recall@5", "recall@10", "mrr@10", "ndcg@10", "recall@100", "mrr@100", "ndcg@100"]
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
    baseline = read_jsonl(args.baseline)
    feature_source = read_jsonl(args.feature_source)
    predictions = [(label, read_jsonl(path)) for label, path in parse_named_paths(args.prediction)]

    eval_qids = qids_with_predictions(feature_source)
    for _label, pred_rows in predictions:
        eval_qids &= qids_with_predictions(pred_rows)
    all_qids = set(group_qrels(qrels)) & eval_qids
    subsets = {"overall": all_qids}
    subsets["hard_subset"] = hard_subset_qids(qrels, baseline, top_k=args.baseline_top_k, miss_top_k=args.miss_top_k) & all_qids
    subsets.update({name: qids & all_qids for name, qids in feature_subsets(qrels, feature_source, top_k=args.feature_top_k).items()})

    rows = []
    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "qrels": args.qrels,
        "baseline": args.baseline,
        "feature_source": args.feature_source,
        "subset_sizes": {name: len(qids) for name, qids in subsets.items()},
        "metrics": defaultdict(dict),
    }
    for subset_name, qids in subsets.items():
        subset_qrels = filter_qrels(qrels, qids)
        for label, pred_rows in predictions:
            metrics = evaluate_retrieval(subset_qrels, filter_predictions(pred_rows, qids), sorted(set(args.ks)))
            payload["metrics"][subset_name][label] = metrics
            rows.append(metric_row(subset_name, label, len(qids), metrics))
    payload["metrics"] = {key: dict(value) for key, value in payload["metrics"].items()}
    write_json(args.output_json, payload)
    write_table(args.output_md, rows)
    print({"output_json": args.output_json, "output_md": args.output_md, "subset_sizes": payload["subset_sizes"]})


if __name__ == "__main__":
    main()
