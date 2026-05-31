from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval, group_predictions
from src.rerank.selective_gate import candidate_gate_rules, intervention_for_query
from src.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune KCH v4 selective gate on validation predictions.")
    parser.add_argument("--qrels", default="data/processed/bioasq_qrels.jsonl")
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--output", default="outputs/rerank/kch_v4_gated_validation_top100.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/kch_v4_gate_tuning_metrics.json")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    return parser.parse_args()


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def rerank_with_rule(rows: list[dict[str, Any]], rule: Any, top_k: int) -> list[dict[str, Any]]:
    grouped = group_predictions(rows)
    output: list[dict[str, Any]] = []
    for qid in sorted(grouped):
        items = grouped[qid]
        gate_items = [
            {
                "base_rank": int(row.get("metadata", {}).get("base_rank", row.get("rank", 999999))),
                "features": row.get("metadata", {}).get("features", {}),
            }
            for row in items
        ]
        strength, reason, stats = intervention_for_query(gate_items, rule)
        scored = []
        for row in items:
            metadata = row.get("metadata", {})
            base = float(metadata.get("base_score_norm", 1.0 / (60.0 + float(metadata.get("base_rank", row["rank"])))))
            model = float(metadata.get("model_score_norm", row.get("score", 0.0)))
            blend = float(metadata.get("blend_weight", 0.0))
            model_combo = blend * base + (1.0 - blend) * model
            score = (1.0 - strength) * base + strength * model_combo
            scored.append((score, row, base, model, blend))
        scored.sort(key=lambda pair: (-pair[0], int(pair[1].get("metadata", {}).get("base_rank", pair[1]["rank"])), str(pair[1]["passage_id"])))
        for rank, (score, row, base, model, blend) in enumerate(scored[:top_k], start=1):
            output.append(
                {
                    **row,
                    "rank": rank,
                    "score": float(score),
                    "retriever": "kch_v4_selective_gate_tuned",
                    "metadata": {
                        **row.get("metadata", {}),
                        "base_score_norm": base,
                        "model_score_norm": model,
                        "blend_weight": blend,
                        "intervention_strength": strength,
                        "gate_reason": reason,
                        "final_blend_weight": strength * (1.0 - blend),
                        "gate_stats": stats,
                        "gate_rule": asdict(rule),
                    },
                }
            )
    return output


def main() -> None:
    args = parse_args()
    qrels = read_jsonl(args.qrels)
    candidate_rows = read_jsonl(args.candidate_predictions)
    qids = set(group_predictions(candidate_rows))
    eval_qrels = filter_qrels(qrels, qids)
    ks = sorted(set(args.ks))

    trials = []
    best = None
    for rule in candidate_gate_rules():
        predictions = rerank_with_rule(candidate_rows, rule, args.top_k)
        metrics = evaluate_retrieval(eval_qrels, predictions, ks)
        key = (
            float(metrics.get("ndcg@10", 0.0)),
            float(metrics.get("mrr@10", 0.0)),
            float(metrics.get("recall@10", 0.0)),
            float(metrics.get("map@10", 0.0)),
        )
        trial = {"rule": asdict(rule), "metrics": metrics, "selection_key": list(key)}
        trials.append(trial)
        if best is None or key > best[0]:
            best = (key, rule, trial, predictions)
    if best is None:
        raise ValueError("No gate rule trials completed.")
    write_jsonl(args.output, best[3])
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "qrels": args.qrels,
        "candidate_predictions": args.candidate_predictions,
        "output": args.output,
        "selected": best[2],
        "top_trials": sorted(trials, key=lambda row: tuple(-float(v) for v in row["selection_key"]))[:10],
    }
    write_json(args.metrics_output, payload)
    print({"output": args.output, "metrics": args.metrics_output, "selected_rule": best[2]["rule"]})


if __name__ == "__main__":
    main()
