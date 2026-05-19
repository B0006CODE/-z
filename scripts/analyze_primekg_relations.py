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

from src.evaluation.retrieval_metrics import group_qrels
from src.knowledge.relations import relation_adjacency
from src.rerank.hypergraph import entity_map
from src.utils import load_config, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PrimeKG relation coverage for retrieved candidates.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictions", default="outputs/retrieval/hybrid_full_top100.jsonl")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--relations", default=None)
    parser.add_argument("--output", default="results/metrics/primekg_relation_stats.json")
    parser.add_argument("--top-m", type=int, default=100)
    return parser.parse_args()


def group_predictions(rows: list[dict[str, Any]], top_m: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    for qid, q_rows in grouped.items():
        q_rows.sort(key=lambda item: int(item["rank"]))
        grouped[qid] = q_rows[:top_m]
    return dict(grouped)


def entity_ids(entity_rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("entity_id", "")) for row in entity_rows if row.get("entity_id")}


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


def summarize(values: list[tuple[int, float]]) -> dict[str, Any]:
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


def relation_features(q_entities: set[str], p_entities: set[str], adjacency: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    related = set()
    relation_count = 0
    for q_entity in q_entities:
        for relation in adjacency.get(q_entity, []):
            target = str(relation.get("target_entity_id", ""))
            if target in p_entities:
                related.add(target)
                relation_count += 1
    return {
        "primekg_relation_count": float(relation_count),
        "primekg_related_passage_entities": float(len(related)),
        "question_relation_coverage": len(related) / len(q_entities) if q_entities else 0.0,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    qrels_path = args.qrels or paths["qrels"]
    question_entities_path = args.question_entities or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    passage_entities_path = args.passage_entities or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")
    relations_path = args.relations or paths.get("primekg_relations", "data/external_knowledge/primekg_project_relations.jsonl")

    qrels_by_qid = group_qrels(read_jsonl(qrels_path))
    preds_by_qid = group_predictions(read_jsonl(args.predictions), args.top_m)
    question_entities = entity_map(read_jsonl(question_entities_path), "question_id")
    passage_entities = entity_map(read_jsonl(passage_entities_path), "passage_id")
    relations = read_jsonl(relations_path) if Path(relations_path).exists() else []
    adjacency = relation_adjacency(relations)

    values: dict[str, list[tuple[int, float]]] = defaultdict(list)
    questions_with_relation = 0
    candidate_rows_with_relation = 0
    for qid, rows in preds_by_qid.items():
        gold = set(qrels_by_qid.get(qid, {}))
        q_entities = entity_ids(question_entities.get(qid, []))
        any_relation = False
        for row in rows:
            pid = str(row["passage_id"])
            p_entities = entity_ids(passage_entities.get(pid, []))
            features = relation_features(q_entities, p_entities, adjacency)
            label = 1 if pid in gold else 0
            if features["primekg_relation_count"] > 0:
                any_relation = True
                candidate_rows_with_relation += 1
            for name, value in features.items():
                values[name].append((label, value))
        if any_relation:
            questions_with_relation += 1

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "predictions": args.predictions,
        "qrels": qrels_path,
        "question_entities": question_entities_path,
        "passage_entities": passage_entities_path,
        "relations": relations_path,
        "top_m": args.top_m,
        "num_questions": len(preds_by_qid),
        "num_relation_rows": len(relations),
        "num_entities_with_relations": len(adjacency),
        "questions_with_any_candidate_relation": questions_with_relation,
        "candidate_rows_with_relation": candidate_rows_with_relation,
        "feature_separability": {name: summarize(feature_values) for name, feature_values in sorted(values.items())},
    }
    write_json(args.output, summary)
    print(
        {
            "output": args.output,
            "num_relation_rows": len(relations),
            "questions_with_any_candidate_relation": questions_with_relation,
            "candidate_rows_with_relation": candidate_rows_with_relation,
        }
    )


if __name__ == "__main__":
    main()
