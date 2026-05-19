from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure dictionary entity overlap sparsity in retrieved candidates.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--predictions", default="outputs/retrieval/hybrid_full_top100.jsonl")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--top-ms", type=int, nargs="+", default=[20, 50, 100])
    return parser.parse_args()


def entity_sets(rows: list[dict[str, Any]], id_key: str) -> dict[str, set[str]]:
    return {
        str(row[id_key]): {entity["entity_id"] for entity in row.get("entities", [])}
        for row in rows
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    qrels_path = args.qrels or paths["qrels"]
    question_entities_path = args.question_entities or paths.get(
        "question_entities", "data/processed/bioasq_question_entities.jsonl"
    )
    passage_entities_path = args.passage_entities or paths.get(
        "passage_entities", "data/processed/bioasq_passage_entities.jsonl"
    )
    output_path = args.output or paths.get("entity_overlap_stats", "results/metrics/entity_overlap_stats.json")

    predictions = read_jsonl(args.predictions)
    qrels = read_jsonl(qrels_path)
    question_entities = entity_sets(read_jsonl(question_entities_path), "question_id")
    passage_entities = entity_sets(read_jsonl(passage_entities_path), "passage_id")

    preds_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        preds_by_qid[str(row["question_id"])].append(row)
    for rows in preds_by_qid.values():
        rows.sort(key=lambda item: int(item["rank"]))

    gold_by_qid: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        gold_by_qid[str(row["question_id"])].add(str(row["passage_id"]))

    top_m_stats = {}
    for top_m in args.top_ms:
        total_candidates = 0
        candidates_with_overlap = 0
        total_gold_retrieved = 0
        gold_retrieved_with_overlap = 0
        questions_with_any_overlap = 0

        for qid, ranked in preds_by_qid.items():
            q_entities = question_entities.get(qid, set())
            any_overlap = False
            for row in ranked[:top_m]:
                pid = str(row["passage_id"])
                overlap = q_entities & passage_entities.get(pid, set())
                total_candidates += 1
                if overlap:
                    candidates_with_overlap += 1
                    any_overlap = True
                if pid in gold_by_qid.get(qid, set()):
                    total_gold_retrieved += 1
                    if overlap:
                        gold_retrieved_with_overlap += 1
            if any_overlap:
                questions_with_any_overlap += 1

        num_questions = len(preds_by_qid)
        top_m_stats[str(top_m)] = {
            "num_questions": num_questions,
            "total_candidates": total_candidates,
            "candidate_overlap_rate": candidates_with_overlap / total_candidates if total_candidates else 0.0,
            "question_any_overlap_rate": questions_with_any_overlap / num_questions if num_questions else 0.0,
            "total_gold_retrieved": total_gold_retrieved,
            "gold_retrieved_overlap_rate": (
                gold_retrieved_with_overlap / total_gold_retrieved if total_gold_retrieved else 0.0
            ),
        }

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "predictions": args.predictions,
        "qrels": qrels_path,
        "question_entities": question_entities_path,
        "passage_entities": passage_entities_path,
        "top_m_stats": top_m_stats,
        "primekg_status": "not_loaded_no_local_source_configured",
        "mesh_status": "dictionary_concepts_only_no_external_mesh_mapping",
    }
    write_json(output_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
