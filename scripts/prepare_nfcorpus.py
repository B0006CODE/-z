from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BEIR NFCorpus in the local retrieval JSONL format.")
    parser.add_argument("--dataset", default="BeIR/nfcorpus", help="Hugging Face BEIR corpus/query dataset.")
    parser.add_argument("--qrels-dataset", default="BeIR/nfcorpus-qrels", help="Hugging Face BEIR qrels dataset.")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for normalized files.")
    parser.add_argument("--prefix", default="nfcorpus", help="Output file prefix.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional number of test queries for a sanity subset.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_corpus(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corpus = []
    for row in rows:
        corpus.append(
            {
                "passage_id": str(row["_id"]),
                "title": row.get("title", "") or "",
                "text": row.get("text", "") or "",
                "metadata": {
                    "source_dataset": "BeIR/nfcorpus",
                    "source_id": str(row["_id"]),
                },
            }
        )
    return corpus


def normalize_queries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = []
    for row in rows:
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        question = text if not title else f"{title} {text}".strip()
        questions.append(
            {
                "question_id": str(row["_id"]),
                "question": question,
                "answer": "",
                "metadata": {
                    "source_dataset": "BeIR/nfcorpus",
                    "source_id": str(row["_id"]),
                    "title": title,
                },
            }
        )
    return questions


def normalize_qrels(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    qrels = []
    for row in rows:
        relevance = float(row.get("score", 1))
        if relevance <= 0:
            continue
        qrels.append(
            {
                "question_id": str(row["query-id"]),
                "passage_id": str(row["corpus-id"]),
                "relevance": relevance,
                "metadata": {
                    "source_dataset": "BeIR/nfcorpus-qrels",
                    "split": split,
                },
            }
        )
    return qrels


def filter_questions_with_qrels(questions: list[dict[str, Any]], qrels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qids = {str(row["question_id"]) for row in qrels}
    return [row for row in questions if str(row["question_id"]) in qids]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_ds = load_dataset(args.dataset, "corpus")["corpus"]
    queries_ds = load_dataset(args.dataset, "queries")["queries"]
    qrels_ds = load_dataset(args.qrels_dataset)

    corpus = normalize_corpus(list(corpus_ds))
    all_questions = normalize_queries(list(queries_ds))
    qrels_by_split = {split: normalize_qrels(list(rows), split) for split, rows in qrels_ds.items()}
    all_qrels = [row for split_rows in qrels_by_split.values() for row in split_rows]
    questions = filter_questions_with_qrels(all_questions, all_qrels)

    write_jsonl(output_dir / f"{args.prefix}_corpus.jsonl", corpus)
    write_jsonl(output_dir / f"{args.prefix}_questions.jsonl", questions)
    write_jsonl(output_dir / f"{args.prefix}_qrels_all.jsonl", all_qrels)
    for split, qrels in qrels_by_split.items():
        write_jsonl(output_dir / f"{args.prefix}_qrels_{split}.jsonl", qrels)
        split_questions = filter_questions_with_qrels(all_questions, qrels)
        write_jsonl(output_dir / f"{args.prefix}_questions_{split}.jsonl", split_questions)

    sample_summary: dict[str, Any] | None = None
    if args.sample_size is not None:
        test_qrels = qrels_by_split.get("test", [])
        test_qids = sorted({str(row["question_id"]) for row in test_qrels})[: args.sample_size]
        sample_qids = set(test_qids)
        sample_questions = [row for row in all_questions if str(row["question_id"]) in sample_qids]
        sample_qrels = [row for row in test_qrels if str(row["question_id"]) in sample_qids]
        write_jsonl(output_dir / f"{args.prefix}_sample_questions.jsonl", sample_questions)
        write_jsonl(output_dir / f"{args.prefix}_sample_qrels.jsonl", sample_qrels)
        sample_summary = {
            "sample_size": args.sample_size,
            "sample_questions": len(sample_questions),
            "sample_qrels": len(sample_qrels),
        }

    split_counts = {
        split: {
            "qrels": len(qrels),
            "questions": len({str(row["question_id"]) for row in qrels}),
            "relevance_distribution": dict(Counter(str(row["relevance"]) for row in qrels)),
        }
        for split, qrels in qrels_by_split.items()
    }
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "qrels_dataset": args.qrels_dataset,
        "license_note": "BeIR/nfcorpus is distributed on Hugging Face with CC-BY-SA-4.0 metadata.",
        "num_corpus_passages": len(corpus),
        "num_queries_with_qrels": len(questions),
        "num_all_qrels": len(all_qrels),
        "split_counts": split_counts,
        "sample": sample_summary,
        "outputs": {
            "corpus": str(output_dir / f"{args.prefix}_corpus.jsonl"),
            "questions": str(output_dir / f"{args.prefix}_questions.jsonl"),
            "qrels_all": str(output_dir / f"{args.prefix}_qrels_all.jsonl"),
        },
    }
    write_json(Path("logs") / f"prepare_{args.prefix}_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
