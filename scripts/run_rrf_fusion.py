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


def parse_source(raw: str) -> tuple[str, float, str]:
    parts = raw.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Source must be formatted as label=weight=path.")
    label, weight_raw, path = parts
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Source label cannot be empty.")
    return label, float(weight_raw), path.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse any number of retrieval/rerank runs with weighted RRF.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=parse_source,
        help="Input run as label=weight=path. Repeat for multiple runs.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser.parse_args()


def add_rows(
    fused: dict[str, dict[str, dict[str, Any]]],
    rows: list[dict[str, Any]],
    *,
    label: str,
    weight: float,
    rrf_k: int,
) -> None:
    for row in rows:
        qid = str(row["question_id"])
        pid = str(row["passage_id"])
        rank = int(row["rank"])
        entry = fused[qid].setdefault(
            pid,
            {
                "question_id": row["question_id"],
                "passage_id": row["passage_id"],
                "score": 0.0,
                "source_ranks": {},
                "source_scores": {},
                "source_retrievers": {},
            },
        )
        entry["score"] += weight / (rrf_k + rank)
        entry["source_ranks"][label] = rank
        entry["source_scores"][label] = float(row.get("score", 0.0))
        entry["source_retrievers"][label] = row.get("retriever", label)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    top_k = args.top_k or int(config["retrieval"].get("top_k", 100))

    fused: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    source_summaries = []
    for label, weight, path in args.source:
        rows = read_jsonl(path)
        add_rows(fused, rows, label=label, weight=weight, rrf_k=args.rrf_k)
        source_summaries.append({"label": label, "weight": weight, "path": path, "num_rows": len(rows)})

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
                    "retriever": "weighted_rrf_fusion",
                    "metadata": {
                        "top_k": top_k,
                        "rrf_k": args.rrf_k,
                        "source_ranks": row["source_ranks"],
                        "source_scores": row["source_scores"],
                        "source_retrievers": row["source_retrievers"],
                    },
                }
            )
    write_jsonl(args.output, predictions)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "sources": source_summaries,
        "output": args.output,
        "num_questions": len(fused),
        "top_k": top_k,
        "rrf_k": args.rrf_k,
        "num_predictions": len(predictions),
    }
    write_json(Path(config["paths"].get("logs_dir", "logs")) / "run_rrf_fusion_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
