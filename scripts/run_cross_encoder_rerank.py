from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.retrieval.dense import passage_text
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank first-stage candidates with a sentence-transformers CrossEncoder.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--questions", default=None, help="Questions JSONL path.")
    parser.add_argument("--corpus", default=None, help="Corpus JSONL path.")
    parser.add_argument("--qrels", default=None, help="Qrels JSONL path for metrics.")
    parser.add_argument("--predictions", default="outputs/retrieval/hybrid_full_top100.jsonl", help="Candidate JSONL path.")
    parser.add_argument("--output", default="outputs/rerank/cross_encoder_test_top100.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/cross_encoder_test_top100_metrics.json")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--top-m", type=int, default=100, help="Number of input candidates per question to score.")
    parser.add_argument("--top-k", type=int, default=100, help="Number of output candidates per question.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit questions for sanity checks.")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    parser.add_argument(
        "--only-predicted-qids",
        action="store_true",
        help="Evaluate only qrels whose question ids appear in the output. Useful for sample sanity checks.",
    )
    return parser.parse_args()


def group_predictions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    for items in grouped.values():
        items.sort(key=lambda row: int(row["rank"]))
    return dict(grouped)


def score_pairs(model: CrossEncoder, pairs: list[list[str]], batch_size: int, show_progress_bar: bool = True) -> list[float]:
    if not pairs:
        return []
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    scores_array = np.asarray(scores, dtype=np.float64)
    if scores_array.ndim > 1:
        scores_array = scores_array[:, -1]
    return [float(score) for score in scores_array.tolist()]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    paths = config["paths"]
    cross_encoder_cfg = config.get("retrieval", {}).get("cross_encoder", {})

    questions_path = args.questions or paths["questions"]
    corpus_path = args.corpus or paths["corpus"]
    qrels_path = args.qrels or paths.get("qrels")
    model_name = args.model_name or cross_encoder_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    batch_size = args.batch_size or int(cross_encoder_cfg.get("batch_size", 32))
    max_length = args.max_length or int(cross_encoder_cfg.get("max_length", 512))

    questions = read_jsonl(questions_path)
    corpus = read_jsonl(corpus_path)
    candidates = read_jsonl(args.predictions)
    question_by_id = {str(row["question_id"]): row for row in questions}
    passage_by_id = {str(row["passage_id"]): row for row in corpus}
    candidates_by_qid = group_predictions(candidates)

    ordered_qids = [str(row["question_id"]) for row in questions if str(row["question_id"]) in candidates_by_qid]
    if args.sample_limit is not None:
        ordered_qids = ordered_qids[: args.sample_limit]

    model_kwargs: dict[str, Any] = {"max_length": max_length}
    if args.device:
        model_kwargs["device"] = args.device
    model = CrossEncoder(model_name, **model_kwargs)

    pair_rows: list[dict[str, Any]] = []
    pairs: list[list[str]] = []
    missing_passages = 0
    for qid in ordered_qids:
        question = question_by_id[qid]["question"]
        rows = candidates_by_qid[qid][: args.top_m]
        for row in rows:
            passage = passage_by_id.get(str(row["passage_id"]))
            if passage is None:
                missing_passages += 1
                continue
            pair_rows.append(row)
            pairs.append([question, passage_text(passage)])

    cross_scores = score_pairs(model, pairs, batch_size=batch_size, show_progress_bar=True)
    scored_by_qid: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for score, row in zip(cross_scores, pair_rows, strict=True):
        scored_by_qid[str(row["question_id"])].append((score, row))

    reranked: list[dict[str, Any]] = []
    for qid in ordered_qids:
        scored = scored_by_qid.get(qid, [])
        scored.sort(key=lambda item: (-item[0], int(item[1]["rank"]), str(item[1]["passage_id"])))
        for rank, (score, row) in enumerate(scored[: args.top_k], start=1):
            reranked.append(
                {
                    "question_id": row["question_id"],
                    "passage_id": row["passage_id"],
                    "rank": rank,
                    "score": float(score),
                    "retriever": "cross_encoder_rerank",
                    "metadata": {
                        "model_name": model_name,
                        "base_rank": int(row["rank"]),
                        "base_score": float(row.get("score", 0.0)),
                        "top_m": args.top_m,
                        "source_metadata": row.get("metadata", {}),
                    },
                }
            )
    write_jsonl(args.output, reranked)

    metrics: dict[str, Any] = {}
    source_metrics: dict[str, Any] = {}
    if qrels_path:
        qrels = read_jsonl(qrels_path)
        if args.only_predicted_qids:
            predicted_qids = {str(row["question_id"]) for row in reranked}
            qrels = [row for row in qrels if str(row["question_id"]) in predicted_qids]
        metrics = evaluate_retrieval(qrels, reranked, sorted(set(args.ks)))
        source_rows = [
            row
            for row in candidates
            if str(row["question_id"]) in {str(item["question_id"]) for item in reranked}
            and int(row["rank"]) <= args.top_k
        ]
        source_metrics = evaluate_retrieval(qrels, source_rows, sorted(set(args.ks)))

    metrics.update(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "questions": questions_path,
            "corpus": corpus_path,
            "qrels": qrels_path,
            "source_predictions": args.predictions,
            "predictions": args.output,
            "model_name": model_name,
            "device": args.device,
            "batch_size": batch_size,
            "max_length": max_length,
            "top_m": args.top_m,
            "top_k": args.top_k,
            "sample_limit": args.sample_limit,
            "num_questions_scored": len(ordered_qids),
            "num_candidates_scored": len(reranked),
            "missing_passages": missing_passages,
            "source_metrics": source_metrics,
        }
    )
    write_json(args.metrics_output, metrics)

    summary = {
        "timestamp": metrics["timestamp"],
        "output": args.output,
        "metrics_output": args.metrics_output,
        "model_name": model_name,
        "num_questions_scored": len(ordered_qids),
        "num_candidates_scored": len(reranked),
        "mrr@10": metrics.get("mrr@10"),
        "recall@10": metrics.get("recall@10"),
        "ndcg@10": metrics.get("ndcg@10"),
        "source_mrr@10": source_metrics.get("mrr@10") if source_metrics else None,
    }
    write_json(Path(paths.get("logs_dir", "logs")) / "run_cross_encoder_rerank_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
