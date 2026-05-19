from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def parse_float_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Grid must contain at least one value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank candidates with PubMed MeSH overlap features.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictions", default="outputs/retrieval/hybrid_full_top100.jsonl")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--output", default="outputs/rerank/mesh_overlap_test_top100.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/mesh_overlap_test_top100_metrics.json")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--mesh-weight", type=float, default=0.0)
    parser.add_argument("--mesh-grid", type=parse_float_grid, default=parse_float_grid("0,0.01,0.02,0.05,0.1,0.2,0.4"))
    parser.add_argument("--tune-weights", action="store_true")
    parser.add_argument("--validation-modulo", type=int, default=5)
    parser.add_argument("--validation-remainder", type=int, default=0)
    parser.add_argument("--target-split", choices=["all", "validation", "test"], default="test")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    return parser.parse_args()


def group_predictions(predictions: list[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["question_id"])].append(row)
    for qid, rows in grouped.items():
        rows.sort(key=lambda item: int(item["rank"]))
        grouped[qid] = rows[:top_k]
    return dict(grouped)


def mesh_map(rows: list[dict[str, Any]], id_key: str) -> dict[str, set[str]]:
    return {
        str(row[id_key]): {str(term["mesh_ui"]) for term in row.get("mesh_terms", []) if term.get("mesh_ui")}
        for row in rows
    }


def qid_bucket(qid: str, modulo: int) -> int:
    if qid.isdigit():
        return int(qid) % modulo
    return sum(ord(char) for char in qid) % modulo


def split_qids(qids: list[str], modulo: int, remainder: int) -> tuple[set[str], set[str]]:
    validation = {qid for qid in qids if qid_bucket(qid, modulo) == remainder}
    return validation, set(qids) - validation


def filter_qrels(qrels: list[dict[str, Any]], keep_qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in keep_qids]


def mesh_features(q_mesh: set[str], p_mesh: set[str]) -> dict[str, float]:
    overlap = q_mesh & p_mesh
    union = q_mesh | p_mesh
    return {
        "mesh_overlap_count": float(len(overlap)),
        "mesh_jaccard": len(overlap) / len(union) if union else 0.0,
        "question_mesh_coverage": len(overlap) / len(q_mesh) if q_mesh else 0.0,
        "passage_mesh_count": float(len(p_mesh)),
    }


def build_rows(
    preds_by_qid: dict[str, list[dict[str, Any]]],
    question_mesh: dict[str, set[str]],
    passage_mesh: dict[str, set[str]],
    *,
    rrf_k: int,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_qid = {}
    for qid, rows in preds_by_qid.items():
        q_mesh = question_mesh.get(qid, set())
        enriched = []
        base_scores = [1.0 / (rrf_k + int(row["rank"])) for row in rows]
        low = min(base_scores) if base_scores else 0.0
        high = max(base_scores) if base_scores else 0.0
        for row, base_score in zip(rows, base_scores, strict=False):
            pid = str(row["passage_id"])
            normalized_base = (base_score - low) / (high - low) if high > low else 1.0
            features = mesh_features(q_mesh, passage_mesh.get(pid, set()))
            features["base_rank_score"] = normalized_base
            enriched.append({"row": row, "base_rank": int(row["rank"]), "features": features})
        rows_by_qid[qid] = enriched
    return rows_by_qid


def rerank(
    rows_by_qid: dict[str, list[dict[str, Any]]],
    *,
    base_weight: float,
    mesh_weight: float,
    top_k: int,
) -> list[dict[str, Any]]:
    predictions = []
    for qid in sorted(rows_by_qid):
        scored = []
        for item in rows_by_qid[qid]:
            features = item["features"]
            score = (
                base_weight * float(features.get("base_rank_score", 0.0))
                + mesh_weight * float(features.get("question_mesh_coverage", 0.0))
            )
            scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["base_rank"], str(pair[1]["row"]["passage_id"])))
        for rank, (score, item) in enumerate(scored[:top_k], start=1):
            row = item["row"]
            predictions.append(
                {
                    "question_id": row["question_id"],
                    "passage_id": row["passage_id"],
                    "rank": rank,
                    "score": float(score),
                    "retriever": "mesh_overlap_rerank",
                    "metadata": {
                        "base_rank": item["base_rank"],
                        "base_weight": base_weight,
                        "mesh_weight": mesh_weight,
                        "features": item["features"],
                        "source_metadata": row.get("metadata", {}),
                    },
                }
            )
    return predictions


def tune_weight(
    rows_by_qid: dict[str, list[dict[str, Any]]],
    qrels: list[dict[str, Any]],
    validation_qids: set[str],
    *,
    base_weight: float,
    mesh_grid: list[float],
    top_k: int,
    ks: list[int],
) -> dict[str, Any]:
    validation_rows = {qid: rows for qid, rows in rows_by_qid.items() if qid in validation_qids}
    validation_qrels = filter_qrels(qrels, validation_qids)
    trials = []
    best = None
    for mesh_weight in mesh_grid:
        predictions = rerank(validation_rows, base_weight=base_weight, mesh_weight=mesh_weight, top_k=top_k)
        metrics = evaluate_retrieval(validation_qrels, predictions, sorted(set(ks)))
        trial = {
            "base_weight": base_weight,
            "mesh_weight": mesh_weight,
            "mrr@10": metrics.get("mrr@10", 0.0),
            "recall@10": metrics.get("recall@10", 0.0),
            "ndcg@10": metrics.get("ndcg@10", 0.0),
        }
        trials.append(trial)
        key = (trial["mrr@10"], trial["recall@10"], trial["ndcg@10"], -mesh_weight)
        if best is None or key > best["key"]:
            best = {"key": key, "trial": trial}
    assert best is not None
    return {
        "selected": best["trial"],
        "trials": sorted(trials, key=lambda row: (-row["mrr@10"], -row["recall@10"], -row["ndcg@10"])),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    paths = config["paths"]
    qrels_path = args.qrels or paths["qrels"]
    question_mesh_path = args.question_mesh or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")
    passage_mesh_path = args.passage_mesh or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")

    qrels = read_jsonl(qrels_path)
    preds_by_qid = group_predictions(read_jsonl(args.predictions), args.top_k)
    question_mesh = mesh_map(read_jsonl(question_mesh_path), "question_id")
    passage_mesh = mesh_map(read_jsonl(passage_mesh_path), "passage_id")
    rows_by_qid = build_rows(preds_by_qid, question_mesh, passage_mesh, rrf_k=args.rrf_k)
    all_qids = sorted(rows_by_qid)
    validation_qids, test_qids = split_qids(all_qids, args.validation_modulo, args.validation_remainder)

    selected = {"base_weight": args.base_weight, "mesh_weight": args.mesh_weight}
    tuning = None
    if args.tune_weights:
        tuning = tune_weight(
            rows_by_qid,
            qrels,
            validation_qids,
            base_weight=args.base_weight,
            mesh_grid=args.mesh_grid,
            top_k=args.top_k,
            ks=args.ks,
        )
        selected = {
            "base_weight": float(tuning["selected"]["base_weight"]),
            "mesh_weight": float(tuning["selected"]["mesh_weight"]),
        }

    if args.target_split == "validation":
        target_qids = validation_qids
    elif args.target_split == "test":
        target_qids = test_qids
    else:
        target_qids = set(all_qids)
    target_rows = {qid: rows for qid, rows in rows_by_qid.items() if qid in target_qids}
    predictions = rerank(target_rows, base_weight=selected["base_weight"], mesh_weight=selected["mesh_weight"], top_k=args.top_k)
    write_jsonl(args.output, predictions)

    target_qrels = filter_qrels(qrels, target_qids)
    metrics = evaluate_retrieval(target_qrels, predictions, sorted(set(args.ks)))
    metrics.update(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "qrels": qrels_path,
            "predictions": args.output,
            "source_predictions": args.predictions,
            "question_mesh": question_mesh_path,
            "passage_mesh": passage_mesh_path,
            "target_split": args.target_split,
            "num_validation_qids": len(validation_qids),
            "num_test_qids": len(test_qids),
            "selected_weights": selected,
            "tuning": tuning,
        }
    )
    write_json(args.metrics_output, metrics)
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "output": args.output,
        "metrics_output": args.metrics_output,
        "target_split": args.target_split,
        "selected_weights": selected,
        "num_questions": len(target_rows),
        "num_predictions": len(predictions),
    }
    write_json(Path(paths["logs_dir"]) / "run_mesh_overlap_rerank_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
