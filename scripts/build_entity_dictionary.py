from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.entities import build_entity_dictionary, candidate_phrases
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local biomedical entity dictionary from BioASQ text.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--corpus", default=None, help="Corpus JSONL path.")
    parser.add_argument("--questions", default=None, help="Questions JSONL path.")
    parser.add_argument("--output", default=None, help="Entity dictionary JSONL path.")
    parser.add_argument("--min-count", type=int, default=None, help="Minimum corpus document frequency.")
    parser.add_argument("--max-terms", type=int, default=None, help="Maximum dictionary terms.")
    parser.add_argument("--max-ngram", type=int, default=None, help="Maximum candidate phrase length.")
    parser.add_argument("--corpus-limit", type=int, default=None, help="Optional corpus record limit for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    paths = config["paths"]
    knowledge_cfg = config.get("knowledge", {}).get("entity_dictionary", {})
    corpus_path = args.corpus or paths["corpus"]
    questions_path = args.questions or paths["questions"]
    output_path = args.output or paths.get("entity_dictionary", "data/processed/entity_dictionary.jsonl")
    min_count = args.min_count or int(knowledge_cfg.get("min_count", 3))
    max_terms = args.max_terms or int(knowledge_cfg.get("max_terms", 50000))
    max_ngram = args.max_ngram or int(knowledge_cfg.get("max_ngram", 5))

    corpus = read_jsonl(corpus_path)
    if args.corpus_limit is not None:
        corpus = corpus[: args.corpus_limit]
    questions = read_jsonl(questions_path)

    required_terms = sorted(
        {
            canonical
            for question in questions
            for canonical, _surface in candidate_phrases(question["question"], max_ngram=max_ngram)
        }
    )
    dictionary = build_entity_dictionary(
        records=corpus,
        text_key="text",
        min_count=min_count,
        max_terms=max_terms,
        max_ngram=max_ngram,
        required_terms=required_terms,
    )
    write_jsonl(output_path, dictionary)

    type_counts = Counter(row["entity_type"] for row in dictionary)
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "corpus": corpus_path,
        "questions": questions_path,
        "output": output_path,
        "min_count": min_count,
        "max_terms": max_terms,
        "max_ngram": max_ngram,
        "num_corpus_records": len(corpus),
        "num_questions": len(questions),
        "num_required_question_terms": len(required_terms),
        "num_dictionary_terms": len(dictionary),
        "entity_type_counts": dict(sorted(type_counts.items())),
    }
    write_json(Path(paths["logs_dir"]) / "build_entity_dictionary_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
