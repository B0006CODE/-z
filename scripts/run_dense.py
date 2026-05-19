from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.dense import DenseRetriever
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense retrieval with a sentence-transformers model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--corpus", default=None, help="Override corpus JSONL path.")
    parser.add_argument("--questions", default=None, help="Override questions JSONL path.")
    parser.add_argument("--output", default=None, help="Override predictions JSONL path.")
    parser.add_argument("--index-path", default=None, help="Override dense index path.")
    parser.add_argument("--model-name", default=None, help="SentenceTransformer model name.")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding batch size.")
    parser.add_argument("--score-batch-size", type=int, default=None, help="Number of queries per dense scoring batch.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of candidates per question.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit questions for sanity runs.")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuilding dense index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    paths = config["paths"]
    dense_cfg = config["retrieval"].get("dense", {})
    corpus_path = args.corpus or paths["corpus"]
    questions_path = args.questions or paths["questions"]
    output_path = args.output or paths["dense_predictions"]
    index_path = args.index_path or paths["dense_index"]
    model_name = args.model_name or dense_cfg.get("model_name", "abhinand/MedEmbed-small-v0.1")
    batch_size = args.batch_size or int(dense_cfg.get("batch_size", 64))
    score_batch_size = args.score_batch_size or int(dense_cfg.get("score_batch_size", 128))
    normalize_embeddings = bool(dense_cfg.get("normalize_embeddings", True))
    top_k = args.top_k or int(config["retrieval"].get("top_k", 100))

    corpus = read_jsonl(corpus_path)
    questions = read_jsonl(questions_path)
    if args.sample_limit is not None:
        questions = questions[: args.sample_limit]

    index_file = Path(index_path)
    if index_file.exists() and not args.rebuild_index:
        retriever = DenseRetriever.load(index_file, batch_size=batch_size, device=args.device)
        index_action = "loaded"
        if retriever.model_name != model_name:
            raise ValueError(
                f"Dense index was built with '{retriever.model_name}', but requested '{model_name}'. "
                "Use --rebuild-index or a different --index-path."
            )
    else:
        retriever = DenseRetriever(
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            device=args.device,
        )
        retriever.fit(corpus)
        retriever.save(index_file)
        index_action = "built"

    predictions = []
    corpus_by_id = {str(row["passage_id"]): row for row in corpus}
    query_texts = [question["question"] for question in questions]
    result_lists = retriever.search_many(query_texts, top_k=top_k, score_batch_size=score_batch_size)
    for question, results in zip(questions, result_lists, strict=True):
        for result in results:
            passage = corpus_by_id.get(str(result["passage_id"]), {})
            predictions.append(
                {
                    "question_id": question["question_id"],
                    "passage_id": result["passage_id"],
                    "rank": result["rank"],
                    "score": result["score"],
                    "retriever": "dense",
                    "metadata": {
                        "top_k": top_k,
                        "model_name": retriever.model_name,
                        "title": passage.get("title", ""),
                    },
                }
            )
    write_jsonl(output_path, predictions)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "index_action": index_action,
        "index_path": str(index_file),
        "model_name": retriever.model_name,
        "batch_size": batch_size,
        "score_batch_size": score_batch_size,
        "device": args.device,
        "num_questions": len(questions),
        "num_corpus_passages": len(corpus),
        "top_k": top_k,
        "num_predictions": len(predictions),
        "output": output_path,
    }
    write_json(Path(paths["logs_dir"]) / "run_dense_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
