from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.medcpt import MedCPTRetriever
from src.utils import load_config, read_jsonl, resolve_torch_device, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedCPT dual-encoder biomedical dense retrieval.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--output", default="outputs/retrieval/medcpt_dense_full_top100.jsonl")
    parser.add_argument("--index-path", default="indexes/dense/bioasq_medcpt_article.npz")
    parser.add_argument("--query-model-name", default="ncbi/MedCPT-Query-Encoder")
    parser.add_argument("--article-model-name", default="ncbi/MedCPT-Article-Encoder")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--query-max-length", type=int, default=64)
    parser.add_argument("--article-max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    paths = config["paths"]

    corpus_path = args.corpus or paths["corpus"]
    questions_path = args.questions or paths["questions"]
    top_k = args.top_k or int(config["retrieval"].get("top_k", 100))
    device = resolve_torch_device(args.device)

    corpus = read_jsonl(corpus_path)
    questions = read_jsonl(questions_path)
    if args.sample_limit is not None:
        questions = questions[: args.sample_limit]

    index_file = Path(args.index_path)
    if index_file.exists() and not args.rebuild_index:
        retriever = MedCPTRetriever.load(index_file, batch_size=args.batch_size, device=device)
        index_action = "loaded"
        if retriever.query_model_name != args.query_model_name or retriever.article_model_name != args.article_model_name:
            raise ValueError("MedCPT index model names do not match requested model names. Use another --index-path or --rebuild-index.")
    else:
        retriever = MedCPTRetriever(
            query_model_name=args.query_model_name,
            article_model_name=args.article_model_name,
            batch_size=args.batch_size,
            query_max_length=args.query_max_length,
            article_max_length=args.article_max_length,
            device=device,
        )
        retriever.fit(corpus)
        retriever.save(index_file)
        index_action = "built"

    corpus_by_id = {str(row["passage_id"]): row for row in corpus}
    result_lists = retriever.search_many(
        [row["question"] for row in questions],
        top_k=top_k,
        score_batch_size=args.score_batch_size,
    )
    predictions = []
    for question, results in zip(questions, result_lists, strict=True):
        for result in results:
            passage = corpus_by_id.get(str(result["passage_id"]), {})
            predictions.append(
                {
                    "question_id": question["question_id"],
                    "passage_id": result["passage_id"],
                    "rank": result["rank"],
                    "score": result["score"],
                    "retriever": "medcpt_dense",
                    "metadata": {
                        "top_k": top_k,
                        "query_model_name": retriever.query_model_name,
                        "article_model_name": retriever.article_model_name,
                        "title": passage.get("title", ""),
                    },
                }
            )
    write_jsonl(args.output, predictions)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "index_action": index_action,
        "index_path": str(index_file),
        "query_model_name": retriever.query_model_name,
        "article_model_name": retriever.article_model_name,
        "device": device,
        "cuda_device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else None,
        "batch_size": args.batch_size,
        "score_batch_size": args.score_batch_size,
        "num_questions": len(questions),
        "num_corpus_passages": len(corpus),
        "top_k": top_k,
        "num_predictions": len(predictions),
        "output": args.output,
    }
    write_json(Path(paths["logs_dir"]) / "run_medcpt_dense_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
