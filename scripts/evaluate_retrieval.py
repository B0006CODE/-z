from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.utils import load_config, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval predictions.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--qrels", default=None, help="Override qrels JSONL path.")
    parser.add_argument("--predictions", default=None, help="Override predictions JSONL path.")
    parser.add_argument("--output", default=None, help="Override metrics JSON path.")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    parser.add_argument(
        "--only-predicted-qids",
        action="store_true",
        help="Evaluate only qrels whose question ids appear in predictions. Use for sample sanity checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    qrels_path = args.qrels or paths["qrels"]
    predictions_path = args.predictions or paths["bm25_predictions"]
    output_path = args.output or paths["bm25_metrics"]

    qrels = read_jsonl(qrels_path)
    predictions = read_jsonl(predictions_path)
    if args.only_predicted_qids:
        predicted_qids = {str(row["question_id"]) for row in predictions}
        qrels = [row for row in qrels if str(row["question_id"]) in predicted_qids]
    metrics = evaluate_retrieval(qrels, predictions, sorted(set(args.ks)))
    metrics.update(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "qrels": qrels_path,
            "predictions": predictions_path,
            "output": output_path,
            "only_predicted_qids": args.only_predicted_qids,
        }
    )
    write_json(output_path, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
