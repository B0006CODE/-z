from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.entities import build_match_index, match_entities_from_index
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract local dictionary entities from questions and passages.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--dictionary", default=None, help="Entity dictionary JSONL path.")
    parser.add_argument("--questions", default=None, help="Questions JSONL path.")
    parser.add_argument("--corpus", default=None, help="Corpus JSONL path.")
    parser.add_argument("--question-output", default=None, help="Question entity JSONL path.")
    parser.add_argument("--passage-output", default=None, help="Passage entity JSONL path.")
    parser.add_argument("--stats-output", default=None, help="Entity feature stats JSON path.")
    parser.add_argument("--max-matches", type=int, default=None, help="Maximum entities per record.")
    parser.add_argument("--corpus-limit", type=int, default=None, help="Optional corpus record limit for debugging.")
    return parser.parse_args()


def summarize(records: list[dict[str, Any]], id_key: str) -> dict[str, Any]:
    counts = [len(record["entities"]) for record in records]
    type_counts: Counter[str] = Counter()
    unique_entities: set[str] = set()
    for record in records:
        for entity in record["entities"]:
            type_counts[entity["entity_type"]] += 1
            unique_entities.add(entity["entity_id"])
    total = len(records)
    with_entities = sum(1 for count in counts if count > 0)
    return {
        "id_key": id_key,
        "num_records": total,
        "records_with_entities": with_entities,
        "record_coverage": with_entities / total if total else 0.0,
        "avg_entities_per_record": sum(counts) / total if total else 0.0,
        "max_entities_per_record": max(counts) if counts else 0,
        "num_unique_entities": len(unique_entities),
        "entity_type_mentions": dict(sorted(type_counts.items())),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    paths = config["paths"]
    knowledge_cfg = config.get("knowledge", {}).get("entity_extraction", {})
    dictionary_path = args.dictionary or paths.get("entity_dictionary", "data/processed/entity_dictionary.jsonl")
    questions_path = args.questions or paths["questions"]
    corpus_path = args.corpus or paths["corpus"]
    question_output = args.question_output or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    passage_output = args.passage_output or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")
    stats_output = args.stats_output or paths.get("entity_stats", "results/metrics/entity_feature_stats.json")
    max_matches = args.max_matches or int(knowledge_cfg.get("max_matches", 128))

    dictionary = read_jsonl(dictionary_path)
    match_index = build_match_index(dictionary)
    questions = read_jsonl(questions_path)
    corpus = read_jsonl(corpus_path)
    if args.corpus_limit is not None:
        corpus = corpus[: args.corpus_limit]

    question_records = []
    for row in questions:
        question_records.append(
            {
                "question_id": row["question_id"],
                "entities": match_entities_from_index(row["question"], match_index, max_matches=max_matches),
            }
        )

    passage_records = []
    for row in corpus:
        passage_records.append(
            {
                "passage_id": row["passage_id"],
                "entities": match_entities_from_index(row["text"], match_index, max_matches=max_matches),
            }
        )

    write_jsonl(question_output, question_records)
    write_jsonl(passage_output, passage_records)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dictionary": dictionary_path,
        "num_dictionary_terms": len(dictionary),
        "questions": questions_path,
        "corpus": corpus_path,
        "question_output": question_output,
        "passage_output": passage_output,
        "max_matches": max_matches,
        "question_entities": summarize(question_records, "question_id"),
        "passage_entities": summarize(passage_records, "passage_id"),
    }
    write_json(stats_output, summary)
    write_json(Path(paths["logs_dir"]) / "extract_entities_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
