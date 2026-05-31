from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def group_qrels(qrels: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in qrels:
        grouped[str(row["question_id"])][str(row["passage_id"])] = float(row.get("relevance", 1))
    return dict(grouped)


def group_predictions(predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["question_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["rank"]))
    return dict(grouped)


def dcg(relevances: list[float]) -> float:
    return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def evaluate_retrieval(
    qrels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    ks: list[int],
) -> dict[str, Any]:
    qrels_by_qid = group_qrels(qrels)
    preds_by_qid = group_predictions(predictions)
    qids = sorted(qrels_by_qid)

    metrics: dict[str, Any] = {
        "num_questions_with_qrels": len(qids),
        "num_questions_with_predictions": len(preds_by_qid),
        "ks": ks,
    }
    if not qids:
        metrics["warning"] = "No qrels found; retrieval metrics cannot be computed."
        return metrics

    for k in ks:
        recall_values: list[float] = []
        precision_values: list[float] = []
        hit_values: list[float] = []
        reciprocal_ranks: list[float] = []
        ndcg_values: list[float] = []
        average_precision_values: list[float] = []

        for qid in qids:
            gold = qrels_by_qid[qid]
            gold_ids = set(gold)
            ranked = preds_by_qid.get(qid, [])[:k]
            retrieved_ids = [str(row["passage_id"]) for row in ranked]
            retrieved_set = set(retrieved_ids)
            hits = gold_ids & retrieved_set

            recall_values.append(len(hits) / len(gold_ids) if gold_ids else 0.0)
            precision_values.append(len(hits) / k if k else 0.0)
            hit_values.append(1.0 if hits else 0.0)

            rr = 0.0
            for rank, passage_id in enumerate(retrieved_ids, start=1):
                if passage_id in gold_ids:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            gains = [gold.get(pid, 0.0) for pid in retrieved_ids]
            ideal_gains = sorted(gold.values(), reverse=True)[:k]
            ideal = dcg(ideal_gains)
            ndcg_values.append(dcg(gains) / ideal if ideal > 0 else 0.0)

            precisions: list[float] = []
            running_hits = 0
            for rank, passage_id in enumerate(retrieved_ids, start=1):
                if passage_id in gold_ids:
                    running_hits += 1
                    precisions.append(running_hits / rank)
            denominator = min(len(gold_ids), k)
            average_precision_values.append(sum(precisions) / denominator if denominator else 0.0)

        metrics[f"recall@{k}"] = sum(recall_values) / len(recall_values)
        metrics[f"precision@{k}"] = sum(precision_values) / len(precision_values)
        metrics[f"hit@{k}"] = sum(hit_values) / len(hit_values)
        metrics[f"mrr@{k}"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
        metrics[f"ndcg@{k}"] = sum(ndcg_values) / len(ndcg_values)
        metrics[f"map@{k}"] = sum(average_precision_values) / len(average_precision_values)

    return metrics
