from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bioasq import choose_split, dataset_schema, normalize_corpus, normalize_questions
from src.utils import load_config, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare rag-mini-bioasq for retrieval experiments.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--sample-size", type=int, default=None, help="Limit questions for sanity runs.")
    parser.add_argument("--split", default=None, help="Override dataset split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    dataset_cfg = config["dataset"]
    paths = config["paths"]
    dataset_name = dataset_cfg["name"]
    requested_split = args.split if args.split is not None else dataset_cfg.get("split")

    available_configs = get_dataset_config_names(dataset_name)
    qa_config = dataset_cfg.get("qa_config")
    corpus_config = dataset_cfg.get("corpus_config")
    if qa_config not in available_configs:
        raise ValueError(f"QA config '{qa_config}' not found. Available configs: {available_configs}")
    if corpus_config not in available_configs:
        raise ValueError(f"Corpus config '{corpus_config}' not found. Available configs: {available_configs}")

    qa_dataset = load_dataset(dataset_name, qa_config)
    corpus_dataset = load_dataset(dataset_name, corpus_config)
    qa_split = choose_split(qa_dataset, requested_split)
    corpus_split = choose_split(corpus_dataset, requested_split)

    qa_rows = [dict(row) for row in qa_dataset[qa_split]]
    corpus_rows = [dict(row) for row in corpus_dataset[corpus_split]]

    corpus = normalize_corpus(corpus_rows)
    questions, qrels = normalize_questions(qa_rows, corpus, sample_size=args.sample_size)

    write_jsonl(paths["corpus"], corpus)
    write_jsonl(paths["questions"], questions)
    write_jsonl(paths["qrels"], qrels)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": dataset_name,
        "available_configs": available_configs,
        "qa_config": qa_config,
        "qa_split": qa_split,
        "corpus_config": corpus_config,
        "corpus_split": corpus_split,
        "sample_size": args.sample_size,
        "num_raw_questions": len(qa_rows),
        "num_raw_corpus_rows": len(corpus_rows),
        "num_processed_questions": len(questions),
        "num_processed_passages": len(corpus),
        "num_qrels": len(qrels),
        "num_questions_with_qrels": len({row["question_id"] for row in qrels}),
        "qa_schema": dataset_schema(qa_rows),
        "corpus_schema": dataset_schema(corpus_rows),
        "outputs": {
            "questions": paths["questions"],
            "corpus": paths["corpus"],
            "qrels": paths["qrels"],
        },
    }
    log_path = Path(paths["logs_dir"]) / "prepare_bioasq_summary.json"
    write_json(log_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
