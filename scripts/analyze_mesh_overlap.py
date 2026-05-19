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
from src.utils import load_config, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MeSH overlap separability for retrieved candidates.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictions", default="outputs/retrieval/hybrid_full_top100.jsonl")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--output", default="results/metrics/mesh_overlap_stats.json")
    parser.add_argument("--top-m", type=int, default=100)
    return parser.parse_args()


def mesh_map(rows: list[dict[str, Any]], id_key: str) -> dict[str, set[str]]:
    return {
        str(row[id_key]): {str(term["mesh_ui"]) for term in row.get("mesh_terms", []) if term.get("mesh_ui")}
        for row in rows
    }


def group_predictions(rows: list[dict[str, Any]], top_m: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    for qid, q_rows in grouped.items():
        q_rows.sort(key=lambda item: int(item["rank"]))
        grouped[qid] = q_rows[:top_m]
    return dict(grouped)


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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    qrels_path = args.qrels or paths["qrels"]
    question_mesh_path = args.question_mesh or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")
    passage_mesh_path = args.passage_mesh or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")

    qrels_by_qid = group_qrels(read_jsonl(qrels_path))
    preds_by_qid = group_predictions(read_jsonl(args.predictions), args.top_m)
    question_mesh = mesh_map(read_jsonl(question_mesh_path), "question_id")
    passage_mesh = mesh_map(read_jsonl(passage_mesh_path), "passage_id")

    values: dict[str, list[tuple[int, float]]] = defaultdict(list)
    questions_with_any_overlap = 0
    for qid, rows in preds_by_qid.items():
        gold = set(qrels_by_qid.get(qid, {}))
        q_mesh = question_mesh.get(qid, set())
        any_overlap = False
        for row in rows:
            pid = str(row["passage_id"])
            p_mesh = passage_mesh.get(pid, set())
            overlap = q_mesh & p_mesh
            union = q_mesh | p_mesh
            label = 1 if pid in gold else 0
            features = {
                "mesh_overlap_count": float(len(overlap)),
                "mesh_jaccard": len(overlap) / len(union) if union else 0.0,
                "question_mesh_coverage": len(overlap) / len(q_mesh) if q_mesh else 0.0,
                "passage_mesh_count": float(len(p_mesh)),
            }
            any_overlap = any_overlap or bool(overlap)
            for name, value in features.items():
                values[name].append((label, value))
        if any_overlap:
            questions_with_any_overlap += 1

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "predictions": args.predictions,
        "qrels": qrels_path,
        "question_mesh": question_mesh_path,
        "passage_mesh": passage_mesh_path,
        "top_m": args.top_m,
        "num_questions": len(preds_by_qid),
        "questions_with_mesh": sum(1 for qid in preds_by_qid if question_mesh.get(qid)),
        "questions_with_any_candidate_mesh_overlap": questions_with_any_overlap,
        "feature_separability": {name: summarize(feature_values) for name, feature_values in sorted(values.items())},
    }
    write_json(args.output, summary)
    print(
        {
            "output": args.output,
            "questions_with_mesh": summary["questions_with_mesh"],
            "questions_with_any_candidate_mesh_overlap": questions_with_any_candidate_mesh_overlap,
        }
        if (questions_with_any_candidate_mesh_overlap := questions_with_any_overlap) is not None
        else summary
    )


if __name__ == "__main__":
    main()
