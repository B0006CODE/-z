from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval, group_qrels
from src.rerank.hypergraph import entity_map
from src.utils import load_config, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose reranking upper bound and feature separability.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--predictions", default="outputs/retrieval/hybrid_full_top100.jsonl")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--output", default="results/metrics/rerank_diagnostics_hybrid_top100.json")
    parser.add_argument("--top-m", type=int, default=100)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    parser.add_argument(
        "--only-predicted-qids",
        action="store_true",
        help="Evaluate only qrels whose question ids appear in predictions.",
    )
    return parser.parse_args()


def group_predictions(predictions: list[dict[str, Any]], top_m: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["question_id"])].append(row)
    for qid, rows in grouped.items():
        rows.sort(key=lambda item: int(item["rank"]))
        grouped[qid] = rows[:top_m]
    return dict(grouped)


def oracle_predictions(preds_by_qid: dict[str, list[dict[str, Any]]], qrels_by_qid: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows_out = []
    for qid in sorted(preds_by_qid):
        gold = set(qrels_by_qid.get(qid, {}))
        ranked = sorted(
            preds_by_qid[qid],
            key=lambda row: (str(row["passage_id"]) not in gold, int(row["rank"]), str(row["passage_id"])),
        )
        for rank, row in enumerate(ranked, start=1):
            rows_out.append(
                {
                    "question_id": row["question_id"],
                    "passage_id": row["passage_id"],
                    "rank": rank,
                    "score": float(len(ranked) + 1 - rank),
                    "retriever": "oracle_within_candidates",
                    "metadata": {"base_rank": int(row["rank"])},
                }
            )
    return rows_out


def auc_from_scores(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(zip(scores, labels, strict=False), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    idx = 0
    while idx < len(ordered):
        j = idx + 1
        while j < len(ordered) and ordered[j][0] == ordered[idx][0]:
            j += 1
        avg_rank = (rank + rank + (j - idx) - 1) / 2.0
        rank_sum += avg_rank * sum(label for _, label in ordered[idx:j])
        rank += j - idx
        idx = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def entity_features(q_entities: set[str], p_entities: set[str]) -> dict[str, float]:
    overlap = q_entities & p_entities
    union = q_entities | p_entities
    return {
        "entity_overlap_count": float(len(overlap)),
        "entity_jaccard": len(overlap) / len(union) if union else 0.0,
        "question_entity_coverage": len(overlap) / len(q_entities) if q_entities else 0.0,
        "passage_entity_count": float(len(p_entities)),
    }


def summarize_feature(values: list[tuple[int, float]]) -> dict[str, Any]:
    gold = [value for label, value in values if label == 1]
    non_gold = [value for label, value in values if label == 0]
    labels = [label for label, _ in values]
    scores = [value for _, value in values]
    return {
        "num_gold": len(gold),
        "num_non_gold": len(non_gold),
        "gold_mean": mean(gold) if gold else 0.0,
        "non_gold_mean": mean(non_gold) if non_gold else 0.0,
        "gold_positive_rate": mean([1.0 if value > 0 else 0.0 for value in gold]) if gold else 0.0,
        "non_gold_positive_rate": mean([1.0 if value > 0 else 0.0 for value in non_gold]) if non_gold else 0.0,
        "auc": auc_from_scores(labels, scores),
    }


def rank_diagnostics(preds_by_qid: dict[str, list[dict[str, Any]]], qrels_by_qid: dict[str, dict[str, float]]) -> dict[str, Any]:
    first_gold_ranks = []
    gold_candidate_counts = []
    missed_qids = []
    for qid, gold in qrels_by_qid.items():
        rows = preds_by_qid.get(qid, [])
        rank_by_pid = {str(row["passage_id"]): int(row["rank"]) for row in rows}
        gold_ranks = [rank_by_pid[pid] for pid in gold if pid in rank_by_pid]
        gold_candidate_counts.append(len(gold_ranks))
        if gold_ranks:
            first_gold_ranks.append(min(gold_ranks))
        else:
            missed_qids.append(qid)
    first_gold_ranks_sorted = sorted(first_gold_ranks)
    return {
        "num_questions": len(qrels_by_qid),
        "num_questions_with_any_gold_in_candidates": len(first_gold_ranks),
        "num_questions_without_gold_in_candidates": len(missed_qids),
        "question_candidate_hit_rate": len(first_gold_ranks) / len(qrels_by_qid) if qrels_by_qid else 0.0,
        "avg_gold_candidates_per_question": mean(gold_candidate_counts) if gold_candidate_counts else 0.0,
        "median_first_gold_rank": first_gold_ranks_sorted[len(first_gold_ranks_sorted) // 2] if first_gold_ranks_sorted else None,
        "missed_qid_examples": missed_qids[:20],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    qrels_path = args.qrels or paths["qrels"]
    question_entities_path = args.question_entities or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    passage_entities_path = args.passage_entities or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")

    predictions = read_jsonl(args.predictions)
    qrels = read_jsonl(qrels_path)
    preds_by_qid = group_predictions(predictions, args.top_m)
    if args.only_predicted_qids:
        predicted_qids = set(preds_by_qid)
        qrels = [row for row in qrels if str(row["question_id"]) in predicted_qids]
    qrels_by_qid = group_qrels(qrels)

    question_entities = {
        qid: {str(entity["entity_id"]) for entity in entities}
        for qid, entities in entity_map(read_jsonl(question_entities_path), "question_id").items()
    }
    passage_entities = {
        pid: {str(entity["entity_id"]) for entity in entities}
        for pid, entities in entity_map(read_jsonl(passage_entities_path), "passage_id").items()
    }

    base_rows = [row for rows in preds_by_qid.values() for row in rows]
    base_metrics = evaluate_retrieval(qrels, base_rows, sorted(set(args.ks)))
    oracle_rows = oracle_predictions(preds_by_qid, qrels_by_qid)
    oracle_metrics = evaluate_retrieval(qrels, oracle_rows, sorted(set(args.ks)))

    feature_values: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for qid, rows in preds_by_qid.items():
        gold = set(qrels_by_qid.get(qid, {}))
        q_entities = question_entities.get(qid, set())
        for row in rows:
            pid = str(row["passage_id"])
            label = 1 if pid in gold else 0
            features = entity_features(q_entities, passage_entities.get(pid, set()))
            features["base_rank_reciprocal"] = 1.0 / int(row["rank"])
            for name, value in features.items():
                feature_values[name].append((label, float(value)))

    diagnostics = {
        "timestamp": datetime.now(UTC).isoformat(),
        "predictions": args.predictions,
        "qrels": qrels_path,
        "question_entities": question_entities_path,
        "passage_entities": passage_entities_path,
        "top_m": args.top_m,
        "base_metrics": base_metrics,
        "oracle_metrics_within_top_m": oracle_metrics,
        "oracle_delta": {
            key: oracle_metrics[key] - base_metrics.get(key, 0.0)
            for key in oracle_metrics
            if key.startswith(("recall@", "mrr@", "ndcg@", "hit@"))
        },
        "rank_diagnostics": rank_diagnostics(preds_by_qid, qrels_by_qid),
        "feature_separability": {
            name: summarize_feature(values)
            for name, values in sorted(feature_values.items())
        },
    }
    write_json(args.output, diagnostics)
    print(
        {
            "output": args.output,
            "top_m": args.top_m,
            "base_mrr@10": base_metrics.get("mrr@10"),
            "oracle_mrr@10": oracle_metrics.get("mrr@10"),
            "base_recall@10": base_metrics.get("recall@10"),
            "oracle_recall@10": oracle_metrics.get("recall@10"),
        }
    )


if __name__ == "__main__":
    main()
