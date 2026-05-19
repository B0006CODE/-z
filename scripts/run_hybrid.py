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
    parser = argparse.ArgumentParser(description="Fuse BM25 and dense predictions using Reciprocal Rank Fusion.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--bm25-predictions", default=None, help="BM25 predictions JSONL path.")
    parser.add_argument("--dense-predictions", default=None, help="Dense predictions JSONL path.")
    parser.add_argument("--output", default=None, help="Hybrid predictions JSONL path.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of candidates per question.")
    parser.add_argument("--rrf-k", type=int, default=None, help="RRF rank constant.")
    parser.add_argument("--bm25-weight", type=float, default=None, help="BM25 RRF weight.")
    parser.add_argument("--dense-weight", type=float, default=None, help="Dense RRF weight.")
    return parser.parse_args()


def add_runs(
    fused: dict[str, dict[str, dict[str, Any]]],
    rows: list[dict[str, Any]],
    source: str,
    weight: float,
    rrf_k: int,
) -> None:
    for row in rows:
        qid = str(row["question_id"])
        pid = str(row["passage_id"])
        rank = int(row["rank"])
        score = weight / (rrf_k + rank)
        entry = fused[qid].setdefault(
            pid,
            {
                "question_id": row["question_id"],
                "passage_id": row["passage_id"],
                "score": 0.0,
                "source_ranks": {},
                "source_scores": {},
            },
        )
        entry["score"] += score
        entry["source_ranks"][source] = rank
        entry["source_scores"][source] = float(row.get("score", 0.0))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    paths = config["paths"]
    hybrid_cfg = config["retrieval"].get("hybrid", {})
    bm25_path = args.bm25_predictions or paths["bm25_predictions"]
    dense_path = args.dense_predictions or paths["dense_predictions"]
    output_path = args.output or paths["hybrid_predictions"]
    top_k = args.top_k or int(config["retrieval"].get("top_k", 100))
    rrf_k = args.rrf_k or int(hybrid_cfg.get("rrf_k", 60))
    bm25_weight = args.bm25_weight or float(hybrid_cfg.get("bm25_weight", 1.0))
    dense_weight = args.dense_weight or float(hybrid_cfg.get("dense_weight", 1.0))

    bm25_rows = read_jsonl(bm25_path)
    dense_rows = read_jsonl(dense_path)
    fused: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    add_runs(fused, bm25_rows, "bm25", bm25_weight, rrf_k)
    add_runs(fused, dense_rows, "dense", dense_weight, rrf_k)

    predictions = []
    for qid in sorted(fused):
        ranked = sorted(
            fused[qid].values(),
            key=lambda row: (-float(row["score"]), min(row["source_ranks"].values()), str(row["passage_id"])),
        )[:top_k]
        for rank, row in enumerate(ranked, start=1):
            predictions.append(
                {
                    "question_id": row["question_id"],
                    "passage_id": row["passage_id"],
                    "rank": rank,
                    "score": float(row["score"]),
                    "retriever": "hybrid_rrf",
                    "metadata": {
                        "top_k": top_k,
                        "rrf_k": rrf_k,
                        "bm25_weight": bm25_weight,
                        "dense_weight": dense_weight,
                        "source_ranks": row["source_ranks"],
                        "source_scores": row["source_scores"],
                    },
                }
            )
    write_jsonl(output_path, predictions)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "bm25_predictions": bm25_path,
        "dense_predictions": dense_path,
        "output": output_path,
        "num_bm25_rows": len(bm25_rows),
        "num_dense_rows": len(dense_rows),
        "num_questions": len(fused),
        "top_k": top_k,
        "rrf_k": rrf_k,
        "bm25_weight": bm25_weight,
        "dense_weight": dense_weight,
        "num_predictions": len(predictions),
    }
    write_json(Path(paths["logs_dir"]) / "run_hybrid_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
