from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25 import BM25Retriever
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 retrieval.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--corpus", default=None, help="Override corpus JSONL path.")
    parser.add_argument("--questions", default=None, help="Override questions JSONL path.")
    parser.add_argument("--output", default=None, help="Override predictions JSONL path.")
    parser.add_argument("--index-path", default=None, help="Override BM25 index path.")
    parser.add_argument("--log-output", default=None, help="Override run summary JSON path.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of candidates per question.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit questions for sanity runs.")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuilding BM25 index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    paths = config["paths"]
    bm25_cfg = config["retrieval"].get("bm25", {})
    corpus_path = args.corpus or paths["corpus"]
    questions_path = args.questions or paths["questions"]
    output_path = args.output or paths["bm25_predictions"]
    index_path = args.index_path or paths["bm25_index"]
    top_k = args.top_k or int(config["retrieval"].get("top_k", 100))

    corpus = read_jsonl(corpus_path)
    questions = read_jsonl(questions_path)
    if args.sample_limit is not None:
        questions = questions[: args.sample_limit]

    index_file = Path(index_path)
    if index_file.exists() and not args.rebuild_index:
        retriever = BM25Retriever.load(index_file)
        index_action = "loaded"
    else:
        retriever = BM25Retriever(k1=float(bm25_cfg.get("k1", 1.5)), b=float(bm25_cfg.get("b", 0.75)))
        retriever.fit(corpus)
        retriever.save(index_file)
        index_action = "built"

    predictions = []
    for question in questions:
        results = retriever.search(question["question"], top_k=top_k)
        for result in results:
            predictions.append(
                {
                    "question_id": question["question_id"],
                    "passage_id": result["passage_id"],
                    "rank": result["rank"],
                    "score": result["score"],
                    "retriever": "bm25",
                    "metadata": {
                        "top_k": top_k,
                        "title": result.get("title", ""),
                    },
                }
            )
    write_jsonl(output_path, predictions)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "index_action": index_action,
        "index_path": str(index_file),
        "num_questions": len(questions),
        "num_corpus_passages": len(corpus),
        "top_k": top_k,
        "num_predictions": len(predictions),
        "output": output_path,
    }
    log_path = Path(args.log_output) if args.log_output else Path(paths["logs_dir"]) / "run_bm25_summary.json"
    write_json(log_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
