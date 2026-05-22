from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval, group_qrels
from src.rerank.lambdamart import make_lambdamart_ranker
from src.utils import read_jsonl, set_seed, write_json, write_jsonl


FEATURE_NAMES = [
    "base_rank_score",
    "hybrid_score",
    "bm25_score",
    "dense_score",
    "bm25_rank_score",
    "dense_rank_score",
    "rank_percentile",
    "bm25_present",
    "dense_present",
]


def parse_int_grid(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Grid must contain at least one integer.")
    return values


def parse_float_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Grid must contain at least one float.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an external BEIR NFCorpus retrieval-only LambdaMART diagnostic."
    )
    parser.add_argument("--bm25-predictions", default="outputs/retrieval/nfcorpus_bm25_full_top100.jsonl")
    parser.add_argument("--dense-predictions", default="outputs/retrieval/nfcorpus_dense_full_top100.jsonl")
    parser.add_argument("--hybrid-predictions", default="outputs/retrieval/nfcorpus_hybrid_full_top100.jsonl")
    parser.add_argument("--train-qrels", default="data/processed/nfcorpus_qrels_train.jsonl")
    parser.add_argument("--validation-qrels", default="data/processed/nfcorpus_qrels_validation.jsonl")
    parser.add_argument("--test-qrels", default="data/processed/nfcorpus_qrels_test.jsonl")
    parser.add_argument("--output", default="outputs/rerank/nfcorpus_retrieval_ltr_test_top100.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/nfcorpus_retrieval_diagnostic.json")
    parser.add_argument("--table-md", default="results/tables/nfcorpus_retrieval_diagnostic.md")
    parser.add_argument("--table-csv", default="results/tables/nfcorpus_retrieval_diagnostic.csv")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    parser.add_argument("--num-leaves-grid", type=parse_int_grid, default=parse_int_grid("7,15,31"))
    parser.add_argument("--learning-rate-grid", type=parse_float_grid, default=parse_float_grid("0.03,0.05"))
    parser.add_argument("--n-estimators-grid", type=parse_int_grid, default=parse_int_grid("80,160"))
    parser.add_argument("--blend-grid", type=parse_float_grid, default=parse_float_grid("0,0.1,0.2,0.35"))
    parser.add_argument("--max-qids-per-split", type=int, default=None, help="Optional sanity-run cap per split.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def qids_from_qrels(qrels: list[dict[str, Any]], max_qids: int | None = None) -> set[str]:
    qids = sorted({str(row["question_id"]) for row in qrels})
    if max_qids is not None:
        qids = qids[:max_qids]
    return set(qids)


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def filter_predictions(predictions: list[dict[str, Any]], qids: set[str], top_k: int) -> list[dict[str, Any]]:
    return [row for row in predictions if str(row["question_id"]) in qids and int(row["rank"]) <= top_k]


def group_predictions(predictions: list[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        if int(row["rank"]) <= top_k:
            grouped[str(row["question_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["rank"]))
    return dict(grouped)


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def row_features(row: dict[str, Any], top_k: int, rrf_k: int) -> dict[str, float]:
    metadata = row.get("metadata", {})
    source_ranks = metadata.get("source_ranks", {})
    source_scores = metadata.get("source_scores", {})
    rank = int(row["rank"])
    return {
        "base_rank_score": 1.0 / (rrf_k + rank),
        "hybrid_score": float(row.get("score", 0.0)),
        "bm25_score": float(source_scores.get("bm25", 0.0)),
        "dense_score": float(source_scores.get("dense", 0.0)),
        "bm25_rank_score": 1.0 / (rrf_k + int(source_ranks["bm25"])) if "bm25" in source_ranks else 0.0,
        "dense_rank_score": 1.0 / (rrf_k + int(source_ranks["dense"])) if "dense" in source_ranks else 0.0,
        "rank_percentile": 1.0 - ((rank - 1) / max(top_k - 1, 1)),
        "bm25_present": 1.0 if "bm25" in source_ranks else 0.0,
        "dense_present": 1.0 if "dense" in source_ranks else 0.0,
    }


def matrix_for_qids(
    hybrid_by_qid: dict[str, list[dict[str, Any]]],
    qids: set[str],
    qrels_by_qid: dict[str, dict[str, float]],
    *,
    top_k: int,
    rrf_k: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[dict[str, Any]]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    meta: list[dict[str, Any]] = []
    for qid in sorted(qids):
        candidates = hybrid_by_qid.get(qid, [])[:top_k]
        if not candidates:
            continue
        groups.append(len(candidates))
        gold = qrels_by_qid.get(qid, {})
        for candidate in candidates:
            features = row_features(candidate, top_k, rrf_k)
            pid = str(candidate["passage_id"])
            rows.append([features[name] for name in FEATURE_NAMES])
            labels.append(int(gold.get(pid, 0.0)))
            meta.append({"qid": qid, "passage_id": pid, "row": candidate, "features": features})
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64), groups, meta


def rerank(
    model: Any,
    hybrid_by_qid: dict[str, list[dict[str, Any]]],
    qids: set[str],
    qrels_by_qid: dict[str, dict[str, float]],
    *,
    top_k: int,
    rrf_k: int,
    blend_weight: float,
) -> list[dict[str, Any]]:
    x, _y, _groups, meta = matrix_for_qids(hybrid_by_qid, qids, qrels_by_qid, top_k=top_k, rrf_k=rrf_k)
    model_scores = model.predict(x) if len(meta) else np.asarray([])

    by_qid_model: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    by_qid_base: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for score, meta_row in zip(model_scores, meta, strict=False):
        qid = meta_row["qid"]
        by_qid_model[qid].append((float(score), meta_row))
        by_qid_base[qid].append((float(meta_row["features"]["base_rank_score"]), meta_row))

    predictions: list[dict[str, Any]] = []
    for qid, scored_rows in sorted(by_qid_model.items()):
        model_norm = {
            meta_row["passage_id"]: value
            for value, (_score, meta_row) in zip(minmax([score for score, _ in scored_rows]), scored_rows, strict=False)
        }
        base_rows = by_qid_base[qid]
        base_norm = {
            meta_row["passage_id"]: value
            for value, (_score, meta_row) in zip(minmax([score for score, _ in base_rows]), base_rows, strict=False)
        }
        reranked = []
        for _score, meta_row in scored_rows:
            pid = meta_row["passage_id"]
            score = blend_weight * base_norm[pid] + (1.0 - blend_weight) * model_norm[pid]
            reranked.append((score, meta_row))
        reranked.sort(key=lambda pair: (-pair[0], int(pair[1]["row"]["rank"]), str(pair[1]["passage_id"])))
        for rank, (score, meta_row) in enumerate(reranked[:top_k], start=1):
            source = meta_row["row"]
            predictions.append(
                {
                    "question_id": source["question_id"],
                    "passage_id": source["passage_id"],
                    "rank": rank,
                    "score": float(score),
                    "retriever": "nfcorpus_retrieval_feature_lambdamart",
                    "metadata": {
                        "base_rank": int(source["rank"]),
                        "blend_weight": float(blend_weight),
                        "features": meta_row["features"],
                        "source_metadata": source.get("metadata", {}),
                    },
                }
            )
    return predictions


def train_and_select(
    args: argparse.Namespace,
    hybrid_by_qid: dict[str, list[dict[str, Any]]],
    train_qids: set[str],
    validation_qids: set[str],
    qrels_by_qid: dict[str, dict[str, float]],
    validation_qrels: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    train_x, train_y, train_group, _ = matrix_for_qids(
        hybrid_by_qid,
        train_qids,
        qrels_by_qid,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
    )
    if int(train_y.sum()) <= 0:
        raise ValueError("No positive training labels in the NFCorpus candidate pool.")

    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    for num_leaves in args.num_leaves_grid:
        for learning_rate in args.learning_rate_grid:
            for n_estimators in args.n_estimators_grid:
                model = make_lambdamart_ranker(
                    seed=args.seed,
                    num_leaves=num_leaves,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                )
                model.fit(train_x, train_y, group=train_group)
                for blend_weight in args.blend_grid:
                    val_predictions = rerank(
                        model,
                        hybrid_by_qid,
                        validation_qids,
                        qrels_by_qid,
                        top_k=args.top_k,
                        rrf_k=args.rrf_k,
                        blend_weight=blend_weight,
                    )
                    val_metrics = evaluate_retrieval(validation_qrels, val_predictions, [10])
                    trial = {
                        "num_leaves": int(num_leaves),
                        "learning_rate": float(learning_rate),
                        "n_estimators": int(n_estimators),
                        "blend_weight": float(blend_weight),
                        "validation_mrr@10": float(val_metrics.get("mrr@10", 0.0)),
                        "validation_recall@10": float(val_metrics.get("recall@10", 0.0)),
                        "validation_ndcg@10": float(val_metrics.get("ndcg@10", 0.0)),
                    }
                    trials.append(trial)
                    key = (
                        trial["validation_mrr@10"],
                        trial["validation_recall@10"],
                        trial["validation_ndcg@10"],
                        -trial["blend_weight"],
                    )
                    if best is None or key > best["key"]:
                        best = {"key": key, "trial": trial}

    assert best is not None
    selected = best["trial"]
    final_qids = train_qids | validation_qids
    final_x, final_y, final_group, _ = matrix_for_qids(
        hybrid_by_qid,
        final_qids,
        qrels_by_qid,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
    )
    final_model = make_lambdamart_ranker(
        seed=args.seed,
        num_leaves=int(selected["num_leaves"]),
        learning_rate=float(selected["learning_rate"]),
        n_estimators=int(selected["n_estimators"]),
    )
    final_model.fit(final_x, final_y, group=final_group)
    diagnostics = {
        "feature_names": FEATURE_NAMES,
        "selected": selected,
        "top_trials": sorted(
            trials,
            key=lambda row: (-row["validation_mrr@10"], -row["validation_recall@10"], -row["validation_ndcg@10"]),
        )[:10],
        "train_rows": int(train_x.shape[0]),
        "train_positive_label_sum": int(train_y.sum()),
        "final_train_rows": int(final_x.shape[0]),
        "final_train_positive_label_sum": int(final_y.sum()),
        "feature_importance": [
            {"feature": name, "importance": float(value)}
            for name, value in sorted(
                zip(FEATURE_NAMES, final_model.feature_importances_, strict=False),
                key=lambda pair: float(pair[1]),
                reverse=True,
            )
        ],
    }
    return final_model, diagnostics


def add_evidence_coverage(metrics: dict[str, Any], ks: list[int]) -> dict[str, Any]:
    enriched = dict(metrics)
    for k in ks:
        if f"recall@{k}" in enriched:
            enriched[f"evidence_coverage@{k}"] = enriched[f"recall@{k}"]
    return enriched


def table_row(method: str, metrics: dict[str, Any], source: dict[str, Any] | None = None) -> dict[str, str]:
    row = {"method": method}
    for key in ["recall@5", "recall@10", "mrr@10", "ndcg@10", "evidence_coverage@10", "recall@100"]:
        row[key] = f"{float(metrics.get(key, 0.0)):.4f}" if key in metrics else ""
    if source:
        row["delta_mrr@10"] = f"{float(metrics.get('mrr@10', 0.0)) - float(source.get('mrr@10', 0.0)):+.4f}"
        row["delta_recall@10"] = f"{float(metrics.get('recall@10', 0.0)) - float(source.get('recall@10', 0.0)):+.4f}"
    else:
        row["delta_mrr@10"] = ""
        row["delta_recall@10"] = ""
    return row


def write_table(path_csv: str | Path, path_md: str | Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "method",
        "recall@5",
        "recall@10",
        "mrr@10",
        "ndcg@10",
        "evidence_coverage@10",
        "recall@100",
        "delta_mrr@10",
        "delta_recall@10",
    ]
    Path(path_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(path_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    Path(path_md).parent.mkdir(parents=True, exist_ok=True)
    Path(path_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    bm25_predictions = read_jsonl(args.bm25_predictions)
    dense_predictions = read_jsonl(args.dense_predictions)
    hybrid_predictions = read_jsonl(args.hybrid_predictions)
    train_qrels_all = read_jsonl(args.train_qrels)
    validation_qrels_all = read_jsonl(args.validation_qrels)
    test_qrels_all = read_jsonl(args.test_qrels)

    train_qids = qids_from_qrels(train_qrels_all, args.max_qids_per_split)
    validation_qids = qids_from_qrels(validation_qrels_all, args.max_qids_per_split)
    test_qids = qids_from_qrels(test_qrels_all, args.max_qids_per_split)
    qrels_all = (
        filter_qrels(train_qrels_all, train_qids)
        + filter_qrels(validation_qrels_all, validation_qids)
        + filter_qrels(test_qrels_all, test_qids)
    )
    validation_qrels = filter_qrels(validation_qrels_all, validation_qids)
    test_qrels = filter_qrels(test_qrels_all, test_qids)
    qrels_by_qid = group_qrels(qrels_all)
    hybrid_by_qid = group_predictions(hybrid_predictions, args.top_k)

    model, diagnostics = train_and_select(args, hybrid_by_qid, train_qids, validation_qids, qrels_by_qid, validation_qrels)
    ltr_predictions = rerank(
        model,
        hybrid_by_qid,
        test_qids,
        qrels_by_qid,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        blend_weight=float(diagnostics["selected"]["blend_weight"]),
    )
    write_jsonl(args.output, ltr_predictions)

    ks = sorted(set(args.ks))
    baseline_predictions = {
        "BM25": filter_predictions(bm25_predictions, test_qids, args.top_k),
        "Dense": filter_predictions(dense_predictions, test_qids, args.top_k),
        "Hybrid RRF": filter_predictions(hybrid_predictions, test_qids, args.top_k),
        "Retrieval-feature LambdaMART": ltr_predictions,
    }
    metrics = {
        name: add_evidence_coverage(evaluate_retrieval(test_qrels, rows, ks), ks)
        for name, rows in baseline_predictions.items()
    }
    hybrid_metrics = metrics["Hybrid RRF"]
    rows = [
        table_row("BM25", metrics["BM25"], hybrid_metrics),
        table_row("Dense", metrics["Dense"], hybrid_metrics),
        table_row("Hybrid RRF", hybrid_metrics),
        table_row("Retrieval-feature LambdaMART", metrics["Retrieval-feature LambdaMART"], hybrid_metrics),
    ]
    write_table(args.table_csv, args.table_md, rows)

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": "BEIR NFCorpus",
        "task_note": "External retrieval-only robustness diagnostic; no NFCorpus-specific MeSH, entity, PrimeKG, or hypergraph knowledge features are used.",
        "train_qrels": args.train_qrels,
        "validation_qrels": args.validation_qrels,
        "test_qrels": args.test_qrels,
        "bm25_predictions": args.bm25_predictions,
        "dense_predictions": args.dense_predictions,
        "hybrid_predictions": args.hybrid_predictions,
        "ltr_predictions": args.output,
        "top_k": args.top_k,
        "ks": ks,
        "max_qids_per_split": args.max_qids_per_split,
        "split": {
            "train_qids": len(train_qids),
            "validation_qids": len(validation_qids),
            "test_qids": len(test_qids),
            "test_qrels": len(test_qrels),
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    write_json(args.metrics_output, payload)
    print(
        {
            "metrics_output": args.metrics_output,
            "table_md": args.table_md,
            "ltr_predictions": args.output,
            "hybrid_mrr@10": hybrid_metrics.get("mrr@10"),
            "ltr_mrr@10": metrics["Retrieval-feature LambdaMART"].get("mrr@10"),
            "hybrid_recall@10": hybrid_metrics.get("recall@10"),
            "ltr_recall@10": metrics["Retrieval-feature LambdaMART"].get("recall@10"),
        }
    )


if __name__ == "__main__":
    main()
