from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import dcg, group_predictions, group_qrels
from src.utils import read_jsonl, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap confidence intervals for retrieval metric differences."
    )
    parser.add_argument("--qrels", required=True, help="Qrels JSONL path.")
    parser.add_argument("--baseline-predictions", required=True, help="Baseline predictions JSONL path.")
    parser.add_argument("--candidate-predictions", required=True, help="Candidate predictions JSONL path.")
    parser.add_argument("--dataset", default="retrieval")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--ks", nargs="+", type=int, default=[10])
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["mrr", "recall", "hit", "ndcg"],
        default=["mrr", "recall"],
    )
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def per_query_metric(
    gold: dict[str, float],
    ranked_rows: list[dict[str, Any]],
    metric: str,
    k: int,
) -> float:
    gold_ids = set(gold)
    ranked_ids = [str(row["passage_id"]) for row in ranked_rows[:k]]
    retrieved = set(ranked_ids)
    hits = gold_ids & retrieved

    if metric == "recall":
        return len(hits) / len(gold_ids) if gold_ids else 0.0
    if metric == "hit":
        return 1.0 if hits else 0.0
    if metric == "mrr":
        for rank, passage_id in enumerate(ranked_ids, start=1):
            if passage_id in gold_ids:
                return 1.0 / rank
        return 0.0
    if metric == "ndcg":
        gains = [gold.get(pid, 0.0) for pid in ranked_ids]
        ideal_gains = sorted(gold.values(), reverse=True)[:k]
        ideal = dcg(ideal_gains)
        return dcg(gains) / ideal if ideal > 0 else 0.0
    raise ValueError(f"Unsupported metric: {metric}")


def collect_values(
    qids: list[str],
    qrels_by_qid: dict[str, dict[str, float]],
    predictions_by_qid: dict[str, list[dict[str, Any]]],
    metric: str,
    k: int,
) -> np.ndarray:
    return np.array(
        [
            per_query_metric(qrels_by_qid[qid], predictions_by_qid.get(qid, []), metric=metric, k=k)
            for qid in qids
        ],
        dtype=np.float64,
    )


def paired_bootstrap(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    num_bootstrap: int,
    seed: int,
    alpha: float,
) -> dict[str, float]:
    if baseline_values.shape != candidate_values.shape:
        raise ValueError("Baseline and candidate arrays must have identical shape.")
    if baseline_values.size == 0:
        raise ValueError("No paired questions available for bootstrap.")

    delta_values = candidate_values - baseline_values
    observed_delta = float(np.mean(delta_values))
    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(num_bootstrap, dtype=np.float64)
    n = delta_values.size
    for idx in range(num_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        boot_deltas[idx] = float(np.mean(delta_values[sample_idx]))

    lower = float(np.quantile(boot_deltas, alpha / 2.0))
    upper = float(np.quantile(boot_deltas, 1.0 - alpha / 2.0))
    p_lower = (float(np.sum(boot_deltas <= 0.0)) + 1.0) / (num_bootstrap + 1.0)
    p_upper = (float(np.sum(boot_deltas >= 0.0)) + 1.0) / (num_bootstrap + 1.0)
    p_value = min(1.0, 2.0 * min(p_lower, p_upper))

    return {
        "baseline_mean": float(np.mean(baseline_values)),
        "candidate_mean": float(np.mean(candidate_values)),
        "delta": observed_delta,
        "relative_delta_percent": (
            observed_delta / float(np.mean(baseline_values)) * 100.0
            if float(np.mean(baseline_values)) != 0.0
            else 0.0
        ),
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value_two_sided": p_value,
    }


def markdown_table(payload: dict[str, Any]) -> str:
    rows = payload["results"]
    headers = [
        "Dataset",
        "Baseline",
        "Candidate",
        "Metric",
        "k",
        "Baseline Score",
        "Candidate Score",
        "Delta",
        "95% CI",
        "Rel. Delta",
        "p-value",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    dataset = payload.get("dataset", "retrieval")
    baseline_label = payload.get("baseline_label", "baseline")
    candidate_label = payload.get("candidate_label", "candidate")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    baseline_label,
                    candidate_label,
                    row["metric"],
                    str(row["k"]),
                    f"{row['baseline_mean']:.4f}",
                    f"{row['candidate_mean']:.4f}",
                    f"{row['delta']:+.4f}",
                    f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}]",
                    f"{row['relative_delta_percent']:+.2f}%",
                    f"{row['p_value_two_sided']:.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        f"Paired bootstrap over {payload['num_paired_questions']} questions, "
        f"{payload['num_bootstrap']} resamples, seed={payload['seed']}."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    qrels_by_qid = group_qrels(read_jsonl(args.qrels))
    baseline_by_qid = group_predictions(read_jsonl(args.baseline_predictions))
    candidate_by_qid = group_predictions(read_jsonl(args.candidate_predictions))
    qids = sorted(set(qrels_by_qid) & set(baseline_by_qid) & set(candidate_by_qid))

    results: list[dict[str, Any]] = []
    for metric in args.metrics:
        for k in sorted(set(args.ks)):
            baseline_values = collect_values(qids, qrels_by_qid, baseline_by_qid, metric, k)
            candidate_values = collect_values(qids, qrels_by_qid, candidate_by_qid, metric, k)
            stats = paired_bootstrap(
                baseline_values,
                candidate_values,
                num_bootstrap=args.num_bootstrap,
                seed=args.seed + 1009 * k + 9173 * len(results),
                alpha=args.alpha,
            )
            results.append(
                {
                    "metric": metric,
                    "k": k,
                    **stats,
                }
            )

    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "qrels": args.qrels,
        "baseline_predictions": args.baseline_predictions,
        "candidate_predictions": args.candidate_predictions,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "num_paired_questions": len(qids),
        "num_qrels_questions": len(qrels_by_qid),
        "num_baseline_questions": len(baseline_by_qid),
        "num_candidate_questions": len(candidate_by_qid),
        "num_bootstrap": args.num_bootstrap,
        "seed": args.seed,
        "alpha": args.alpha,
        "results": results,
    }
    write_json(args.output_json, payload)

    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(markdown_table(payload), encoding="utf-8")

    print(
        {
            "output_json": args.output_json,
            "output_md": args.output_md,
            "num_paired_questions": len(qids),
            "results": results,
        }
    )


if __name__ == "__main__":
    main()
