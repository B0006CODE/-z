from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank candidates with dictionary entity overlap via RRF.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--predictions", default="outputs/retrieval/hybrid_full_top100.jsonl")
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--entity-weight", type=float, default=1.0)
    return parser.parse_args()


def entity_sets(rows: list[dict[str, Any]], id_key: str) -> dict[str, set[str]]:
    return {
        str(row[id_key]): {entity["entity_id"] for entity in row.get("entities", [])}
        for row in rows
    }


def overlap_features(question_entities: set[str], passage_entities: set[str]) -> dict[str, float]:
    overlap = question_entities & passage_entities
    union = question_entities | passage_entities
    return {
        "entity_overlap_count": float(len(overlap)),
        "entity_jaccard": len(overlap) / len(union) if union else 0.0,
        "question_entity_coverage": len(overlap) / len(question_entities) if question_entities else 0.0,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    paths = config["paths"]
    question_entities_path = args.question_entities or paths.get(
        "question_entities", "data/processed/bioasq_question_entities.jsonl"
    )
    passage_entities_path = args.passage_entities or paths.get(
        "passage_entities", "data/processed/bioasq_passage_entities.jsonl"
    )
    output_path = args.output or paths.get("entity_overlap_predictions", "outputs/rerank/entity_overlap_full_top100.jsonl")

    predictions = read_jsonl(args.predictions)
    question_entities = entity_sets(read_jsonl(question_entities_path), "question_id")
    passage_entities = entity_sets(read_jsonl(passage_entities_path), "passage_id")

    preds_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        preds_by_qid[str(row["question_id"])].append(row)
    for rows in preds_by_qid.values():
        rows.sort(key=lambda item: int(item["rank"]))

    reranked = []
    for qid in sorted(preds_by_qid):
        rows = preds_by_qid[qid][: args.top_k]
        q_entities = question_entities.get(qid, set())
        enriched = []
        for row in rows:
            p_entities = passage_entities.get(str(row["passage_id"]), set())
            features = overlap_features(q_entities, p_entities)
            enriched.append({"row": row, "features": features})

        entity_ranked = sorted(
            enriched,
            key=lambda item: (
                -item["features"]["entity_overlap_count"],
                -item["features"]["question_entity_coverage"],
                -item["features"]["entity_jaccard"],
                int(item["row"]["rank"]),
            ),
        )
        entity_ranks = {str(item["row"]["passage_id"]): rank for rank, item in enumerate(entity_ranked, start=1)}

        fused = []
        for item in enriched:
            row = item["row"]
            base_rank = int(row["rank"])
            entity_rank = entity_ranks[str(row["passage_id"])]
            score = args.base_weight / (args.rrf_k + base_rank) + args.entity_weight / (args.rrf_k + entity_rank)
            fused.append(
                {
                    "question_id": row["question_id"],
                    "passage_id": row["passage_id"],
                    "score": float(score),
                    "base_rank": base_rank,
                    "entity_rank": entity_rank,
                    "features": item["features"],
                    "source_metadata": row.get("metadata", {}),
                }
            )

        fused.sort(key=lambda item: (-item["score"], item["entity_rank"], item["base_rank"], str(item["passage_id"])))
        for rank, item in enumerate(fused[: args.top_k], start=1):
            reranked.append(
                {
                    "question_id": item["question_id"],
                    "passage_id": item["passage_id"],
                    "rank": rank,
                    "score": item["score"],
                    "retriever": "entity_overlap_rrf",
                    "metadata": {
                        "base_rank": item["base_rank"],
                        "entity_rank": item["entity_rank"],
                        "rrf_k": args.rrf_k,
                        "base_weight": args.base_weight,
                        "entity_weight": args.entity_weight,
                        **item["features"],
                        "source_metadata": item["source_metadata"],
                    },
                }
            )

    write_jsonl(output_path, reranked)
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "predictions": args.predictions,
        "question_entities": question_entities_path,
        "passage_entities": passage_entities_path,
        "output": output_path,
        "top_k": args.top_k,
        "rrf_k": args.rrf_k,
        "base_weight": args.base_weight,
        "entity_weight": args.entity_weight,
        "num_questions": len(preds_by_qid),
        "num_predictions": len(reranked),
    }
    write_json(Path(paths["logs_dir"]) / "run_entity_overlap_rerank_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
