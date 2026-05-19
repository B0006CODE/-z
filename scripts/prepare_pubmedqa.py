from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pubmedqa import normalize_pubmedqa, summarize_pubmedqa
from src.utils import load_config, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare qiaojin/PubMedQA pqa_labeled for retrieval experiments.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--dataset-name", default="qiaojin/PubMedQA")
    parser.add_argument("--dataset-config", default="pqa_labeled")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=None, help="Limit questions for sanity runs.")
    parser.add_argument("--output-prefix", default="data/processed/pubmedqa_pqa_labeled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    available_configs = get_dataset_config_names(args.dataset_name)
    if args.dataset_config not in available_configs:
        raise ValueError(f"Dataset config '{args.dataset_config}' not found. Available: {available_configs}")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    if args.split not in dataset:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(dataset.keys())}")

    raw_rows = [dict(row) for row in dataset[args.split]]
    questions, corpus, qrels, labels, passage_mesh, question_mesh = normalize_pubmedqa(
        raw_rows,
        sample_size=args.sample_size,
        question_prefix="pubmedqa",
    )

    prefix = args.output_prefix
    outputs = {
        "questions": f"{prefix}_questions.jsonl",
        "corpus": f"{prefix}_corpus.jsonl",
        "qrels": f"{prefix}_qrels.jsonl",
        "answer_labels": f"{prefix}_answer_labels.jsonl",
        "passage_mesh": f"{prefix}_passage_mesh.jsonl",
        "question_mesh": f"{prefix}_question_mesh.jsonl",
    }
    write_jsonl(outputs["questions"], questions)
    write_jsonl(outputs["corpus"], corpus)
    write_jsonl(outputs["qrels"], qrels)
    write_jsonl(outputs["answer_labels"], labels)
    write_jsonl(outputs["passage_mesh"], passage_mesh)
    write_jsonl(outputs["question_mesh"], question_mesh)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": args.dataset_name,
        "available_configs": available_configs,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "sample_size": args.sample_size,
        "num_raw_rows": len(raw_rows),
        **summarize_pubmedqa(questions, corpus, qrels, labels, passage_mesh, question_mesh),
        "outputs": outputs,
        "notes": [
            "Each PubMedQA abstract section is a corpus passage.",
            "All sections from the question's source abstract are treated as relevant evidence.",
            "Answer labels are saved separately for later yes/no/maybe QA evaluation.",
        ],
    }
    log_path = Path(config["paths"].get("logs_dir", "logs")) / "prepare_pubmedqa_summary.json"
    write_json(log_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
