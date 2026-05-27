from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import dcg, evaluate_retrieval, group_predictions, group_qrels
from src.utils import read_jsonl, write_json, write_jsonl


FEATURE_NAMES = [
    "max_hypergraph",
    "mean_hypergraph",
    "max_mesh_overlap",
    "max_entity_overlap",
    "max_mesh_hierarchy",
    "max_shared_cluster",
    "mean_base_rank",
    "top10_jaccard_disagreement",
    "mean_abs_rank_delta_top20",
    "no_direct_mesh_overlap",
    "entity_overlap_zero",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight query-level gate between flat and hypergraph LTR outputs.")
    parser.add_argument("--qrels", default="data/processed/bioasq_qrels.jsonl")
    parser.add_argument("--flat-validation", required=True)
    parser.add_argument("--hyper-validation", required=True)
    parser.add_argument("--flat-test", required=True)
    parser.add_argument("--hyper-test", required=True)
    parser.add_argument("--output", default="outputs/rerank/learned_hypergraph_gate_test_top200.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/learned_hypergraph_gate_metrics.json")
    parser.add_argument("--table-output", default="results/tables/learned_hypergraph_gate.md")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--primary", choices=["mrr@10", "recall@10", "ndcg@10"], default="mrr@10")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 100, 200])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def feature(row: dict[str, Any], name: str) -> float:
    return float(row.get("metadata", {}).get("features", {}).get(name, 0.0))


def per_query_scores(qrels_by_qid: dict[str, dict[str, float]], rows_by_qid: dict[str, list[dict[str, Any]]], qid: str, *, k: int = 10) -> dict[str, float]:
    gold = qrels_by_qid.get(qid, {})
    if not gold:
        return {"recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}
    gold_ids = set(gold)
    ranked = rows_by_qid.get(qid, [])[:k]
    retrieved = [str(row["passage_id"]) for row in ranked]
    hits = gold_ids & set(retrieved)
    rr = 0.0
    for rank, pid in enumerate(retrieved, start=1):
        if pid in gold_ids:
            rr = 1.0 / rank
            break
    gains = [gold.get(pid, 0.0) for pid in retrieved]
    ideal = dcg(sorted(gold.values(), reverse=True)[:k])
    return {
        "recall@10": len(hits) / len(gold_ids),
        "mrr@10": rr,
        "ndcg@10": dcg(gains) / ideal if ideal > 0 else 0.0,
    }


def rank_lookup(rows: list[dict[str, Any]], limit: int) -> dict[str, int]:
    return {str(row["passage_id"]): int(row["rank"]) for row in rows[:limit]}


def query_features(flat_rows: list[dict[str, Any]], hyper_rows: list[dict[str, Any]]) -> dict[str, float]:
    top20 = hyper_rows[:20]
    top10 = hyper_rows[:10]
    flat_ranks = rank_lookup(flat_rows, 20)
    hyper_ranks = rank_lookup(hyper_rows, 20)
    union = set(flat_ranks) | set(hyper_ranks)
    rank_delta = [
        abs(float(flat_ranks.get(pid, 21)) - float(hyper_ranks.get(pid, 21)))
        for pid in union
    ]
    flat_top10 = set(rank_lookup(flat_rows, 10))
    hyper_top10 = set(rank_lookup(hyper_rows, 10))
    jaccard = len(flat_top10 & hyper_top10) / len(flat_top10 | hyper_top10) if flat_top10 or hyper_top10 else 1.0
    max_mesh = max((feature(row, "mesh_overlap_count") for row in top20), default=0.0)
    max_entity = max((feature(row, "entity_overlap_count") for row in top20), default=0.0)
    values = {
        "max_hypergraph": max((feature(row, "hypergraph_score_norm") for row in top20), default=0.0),
        "mean_hypergraph": sum(feature(row, "hypergraph_score_norm") for row in top20) / len(top20) if top20 else 0.0,
        "max_mesh_overlap": max_mesh,
        "max_entity_overlap": max_entity,
        "max_mesh_hierarchy": max((feature(row, "question_mesh_hierarchy_coverage") for row in top20), default=0.0),
        "max_shared_cluster": max((feature(row, "shared_mesh_term_cluster_size") for row in top20), default=0.0),
        "mean_base_rank": sum(float(row.get("metadata", {}).get("base_rank") or row["rank"]) for row in top10) / len(top10) if top10 else 0.0,
        "top10_jaccard_disagreement": 1.0 - jaccard,
        "mean_abs_rank_delta_top20": sum(rank_delta) / len(rank_delta) if rank_delta else 0.0,
        "no_direct_mesh_overlap": 1.0 if max_mesh <= 0.0 else 0.0,
        "entity_overlap_zero": 1.0 if max_entity <= 0.0 else 0.0,
    }
    return values


def build_examples(
    qrels_by_qid: dict[str, dict[str, float]],
    flat_rows: list[dict[str, Any]],
    hyper_rows: list[dict[str, Any]],
    primary: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, dict[str, Any]]]:
    flat_by_qid = group_predictions(flat_rows)
    hyper_by_qid = group_predictions(hyper_rows)
    qids = sorted(set(flat_by_qid) & set(hyper_by_qid) & set(qrels_by_qid))
    x_rows = []
    y_rows = []
    detail = {}
    for qid in qids:
        flat_scores = per_query_scores(qrels_by_qid, flat_by_qid, qid)
        hyper_scores = per_query_scores(qrels_by_qid, hyper_by_qid, qid)
        label = 1 if (
            hyper_scores[primary] > flat_scores[primary]
            or (
                hyper_scores[primary] == flat_scores[primary]
                and (
                    hyper_scores["recall@10"],
                    hyper_scores["ndcg@10"],
                    hyper_scores["mrr@10"],
                )
                > (
                    flat_scores["recall@10"],
                    flat_scores["ndcg@10"],
                    flat_scores["mrr@10"],
                )
            )
        ) else 0
        feats = query_features(flat_by_qid[qid], hyper_by_qid[qid])
        x_rows.append([feats[name] for name in FEATURE_NAMES])
        y_rows.append(label)
        detail[qid] = {
            "oracle_label": "hypergraph" if label else "flat",
            "features": feats,
            "flat_scores": flat_scores,
            "hypergraph_scores": hyper_scores,
        }
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64), qids, detail


def train_gate(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[str, Any, dict[str, Any]]:
    if len(set(y.tolist())) < 2:
        constant = int(y[0]) if len(y) else 0
        return "constant", constant, {"constant_class": constant}
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    candidates = [
        ("logistic_regression", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
        ("decision_tree_depth2", DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=seed)),
        ("decision_tree_depth3", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed)),
    ]
    trained = []
    for name, model in candidates:
        model.fit(x, y)
        pred = model.predict(x)
        accuracy = float((pred == y).mean()) if len(y) else 0.0
        trained.append((accuracy, name, model))
    trained.sort(key=lambda item: (-item[0], item[1]))
    accuracy, name, model = trained[0]
    return name, model, {"training_oracle_accuracy": accuracy}


def predict_gate(model_name: str, model: Any, x: np.ndarray) -> np.ndarray:
    if model_name == "constant":
        return np.full((x.shape[0],), int(model), dtype=np.int64)
    return model.predict(x).astype(np.int64)


def apply_gate(
    flat_rows: list[dict[str, Any]],
    hyper_rows: list[dict[str, Any]],
    qids: list[str],
    labels: np.ndarray,
    details: dict[str, dict[str, Any]],
    *,
    top_k: int,
    retriever_name: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    flat_by_qid = group_predictions(flat_rows)
    hyper_by_qid = group_predictions(hyper_rows)
    output = []
    counts: Counter[str] = Counter()
    for qid, label in zip(qids, labels, strict=False):
        selected = "hypergraph" if int(label) == 1 else "flat"
        source = hyper_by_qid[qid] if selected == "hypergraph" else flat_by_qid[qid]
        counts[selected] += 1
        for rank, row in enumerate(source[:top_k], start=1):
            output.append(
                {
                    **row,
                    "rank": rank,
                    "retriever": retriever_name,
                    "metadata": {
                        **row.get("metadata", {}),
                        "gate_selected": selected,
                        "gate_query_features": details[qid]["features"],
                    },
                }
            )
    return output, counts


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def metric_row(method: str, metrics: dict[str, Any]) -> dict[str, str]:
    row = {"method": method}
    for key in ["recall@5", "recall@10", "mrr@10", "ndcg@10", "recall@100", "recall@200"]:
        row[key] = f"{float(metrics.get(key, 0.0)):.4f}" if key in metrics else ""
    return row


def write_table(path: str | Path, rows: list[dict[str, str]]) -> None:
    columns = ["method", "recall@5", "recall@10", "mrr@10", "ndcg@10", "recall@100", "recall@200"]
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
    qrels_by_qid = group_qrels(qrels)
    flat_validation = read_jsonl(args.flat_validation)
    hyper_validation = read_jsonl(args.hyper_validation)
    flat_test = read_jsonl(args.flat_test)
    hyper_test = read_jsonl(args.hyper_test)

    train_x, train_y, validation_qids, validation_details = build_examples(qrels_by_qid, flat_validation, hyper_validation, args.primary)
    model_name, model, model_info = train_gate(train_x, train_y, args.seed)
    validation_pred_labels = predict_gate(model_name, model, train_x)
    validation_gate, validation_counts = apply_gate(
        flat_validation,
        hyper_validation,
        validation_qids,
        validation_pred_labels,
        validation_details,
        top_k=args.top_k,
        retriever_name="learned_hypergraph_gate_validation",
    )

    test_x, test_y_oracle, test_qids, test_details = build_examples(qrels_by_qid, flat_test, hyper_test, args.primary)
    test_pred_labels = predict_gate(model_name, model, test_x)
    test_gate, test_counts = apply_gate(
        flat_test,
        hyper_test,
        test_qids,
        test_pred_labels,
        test_details,
        top_k=args.top_k,
        retriever_name="learned_hypergraph_gate",
    )
    oracle_gate, oracle_counts = apply_gate(
        flat_test,
        hyper_test,
        test_qids,
        test_y_oracle,
        test_details,
        top_k=args.top_k,
        retriever_name="oracle_hypergraph_gate",
    )

    write_jsonl(args.output, test_gate)
    test_qrels = filter_qrels(qrels, set(test_qids))
    validation_qrels = filter_qrels(qrels, set(validation_qids))
    flat_test_metrics = evaluate_retrieval(test_qrels, flat_test, sorted(set(args.ks)))
    hyper_test_metrics = evaluate_retrieval(test_qrels, hyper_test, sorted(set(args.ks)))
    gate_test_metrics = evaluate_retrieval(test_qrels, test_gate, sorted(set(args.ks)))
    oracle_test_metrics = evaluate_retrieval(test_qrels, oracle_gate, sorted(set(args.ks)))
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "primary": args.primary,
        "model_name": model_name,
        "model_info": model_info,
        "feature_names": FEATURE_NAMES,
        "validation_oracle_counts": dict(Counter("hypergraph" if int(label) else "flat" for label in train_y)),
        "validation_gate_counts": dict(validation_counts),
        "test_oracle_counts": dict(Counter("hypergraph" if int(label) else "flat" for label in test_y_oracle)),
        "test_gate_counts": dict(test_counts),
        "test_oracle_accuracy": float((test_pred_labels == test_y_oracle).mean()) if len(test_y_oracle) else 0.0,
        "validation_gate_metrics": evaluate_retrieval(validation_qrels, validation_gate, sorted(set(args.ks))),
        "flat_test_metrics": flat_test_metrics,
        "hypergraph_test_metrics": hyper_test_metrics,
        "learned_gate_test_metrics": gate_test_metrics,
        "oracle_gate_test_metrics": oracle_test_metrics,
        "test_query_details_preview": {qid: test_details[qid] for qid in test_qids[:20]},
        "output": args.output,
    }
    write_json(args.metrics_output, payload)
    write_table(
        args.table_output,
        [
            metric_row("Flat knowledge LTR", flat_test_metrics),
            metric_row("Full hypergraph LTR", hyper_test_metrics),
            metric_row("Learned hypergraph gate", gate_test_metrics),
            metric_row("Oracle hypergraph gate", oracle_test_metrics),
        ],
    )
    print(
        {
            "output": args.output,
            "metrics": args.metrics_output,
            "model": model_name,
            "test_gate_counts": dict(test_counts),
            "learned_mrr@10": gate_test_metrics.get("mrr@10"),
            "oracle_mrr@10": oracle_test_metrics.get("mrr@10"),
        }
    )


if __name__ == "__main__":
    main()
