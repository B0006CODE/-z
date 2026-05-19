from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval, group_qrels
from src.rerank.hypergraph import build_feature_rows, entity_map, mesh_map, relations_map
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


FEATURE_NAMES = [
    "base_rank_score",
    "hybrid_score",
    "bm25_score",
    "dense_score",
    "bm25_rank_score",
    "dense_rank_score",
    "rank_percentile",
    "hypergraph_score_norm",
    "entity_overlap_count",
    "entity_jaccard",
    "question_entity_coverage",
    "passage_entity_count",
    "mesh_overlap_count",
    "mesh_jaccard",
    "question_mesh_coverage",
    "passage_mesh_count",
    "primekg_relation_count",
    "question_relation_coverage",
    "local_num_nodes",
    "local_num_hyperedges",
    "local_shared_entity_edges",
    "local_document_mesh_edges",
    "local_primekg_relation_edges",
]


def parse_float_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Grid must contain at least one float value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight supervised reranker over hybrid and knowledge features.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictions", default="outputs/retrieval/hybrid_full_top100.jsonl")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--relations", default=None)
    parser.add_argument("--output", default="outputs/rerank/learning_rerank_test_top100.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/learning_rerank_test_top100_metrics.json")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--max-passage-entities", type=int, default=48)
    parser.add_argument("--max-passage-mesh", type=int, default=32)
    parser.add_argument("--split-modulo", type=int, default=5)
    parser.add_argument("--validation-remainders", type=int, nargs="+", default=[3])
    parser.add_argument("--test-remainders", type=int, nargs="+", default=[4])
    parser.add_argument("--model", choices=["logreg", "hist_gradient"], default="logreg")
    parser.add_argument("--c-grid", type=parse_float_grid, default=parse_float_grid("0.05,0.1,0.25,0.5,1.0,2.0"))
    parser.add_argument("--blend-grid", type=parse_float_grid, default=parse_float_grid("0,0.1,0.2,0.35,0.5,0.65,0.8,1.0"))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    return parser.parse_args()


def qid_bucket(qid: str, modulo: int) -> int:
    if qid.isdigit():
        return int(qid) % modulo
    return sum(ord(char) for char in qid) % modulo


def split_qids(
    qids: list[str],
    *,
    modulo: int,
    validation_remainders: set[int],
    test_remainders: set[int],
) -> dict[str, set[str]]:
    validation = {qid for qid in qids if qid_bucket(qid, modulo) in validation_remainders}
    test = {qid for qid in qids if qid_bucket(qid, modulo) in test_remainders}
    train = set(qids) - validation - test
    if not train or not validation or not test:
        raise ValueError("Train, validation, and test splits must all be non-empty.")
    return {"train": train, "validation": validation, "test": test}


def source_feature(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    metadata = row.get("metadata", {})
    if name == "bm25_score":
        return float(metadata.get("source_scores", {}).get("bm25", default))
    if name == "dense_score":
        return float(metadata.get("source_scores", {}).get("dense", default))
    if name == "bm25_rank_score":
        rank = metadata.get("source_ranks", {}).get("bm25")
        return 0.0 if rank is None else 1.0 / (60.0 + float(rank))
    if name == "dense_rank_score":
        rank = metadata.get("source_ranks", {}).get("dense")
        return 0.0 if rank is None else 1.0 / (60.0 + float(rank))
    raise KeyError(name)


def enriched_feature_vector(item: dict[str, Any], top_k: int) -> dict[str, float]:
    row = item["row"]
    features = dict(item["features"])
    features["hybrid_score"] = float(row.get("score", 0.0))
    features["bm25_score"] = source_feature(row, "bm25_score")
    features["dense_score"] = source_feature(row, "dense_score")
    features["bm25_rank_score"] = source_feature(row, "bm25_rank_score")
    features["dense_rank_score"] = source_feature(row, "dense_rank_score")
    base_rank = int(item["base_rank"])
    features["rank_percentile"] = 1.0 - ((base_rank - 1) / max(top_k - 1, 1))
    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def build_matrix(
    features_by_qid: dict[str, list[dict[str, Any]]],
    qids: set[str],
    qrels_by_qid: dict[str, dict[str, float]],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []
    for qid in sorted(qids):
        gold = qrels_by_qid.get(qid, {})
        for item in features_by_qid.get(qid, []):
            passage_id = str(item["row"]["passage_id"])
            vector = enriched_feature_vector(item, top_k)
            rows.append([vector[name] for name in FEATURE_NAMES])
            labels.append(1 if passage_id in gold else 0)
            meta.append({"qid": qid, "passage_id": passage_id, "item": item, "features": vector})
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64), meta


def make_model(model_name: str, c_value: float, seed: int) -> Any:
    if model_name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
                solver="liblinear",
            ),
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=15,
        l2_regularization=c_value,
        random_state=seed,
    )


def minmax_by_qid(scores: dict[str, list[tuple[float, dict[str, Any]]]]) -> dict[str, list[tuple[float, dict[str, Any]]]]:
    normalized: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for qid, rows in scores.items():
        values = [score for score, _ in rows]
        low = min(values) if values else 0.0
        high = max(values) if values else 0.0
        if high <= low:
            normalized[qid] = [(1.0, meta) for _, meta in rows]
        else:
            normalized[qid] = [((score - low) / (high - low), meta) for score, meta in rows]
    return normalized


def predict_probabilities(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)[:, 1]
    return model.decision_function(matrix)


def rerank_split(
    model: Any,
    features_by_qid: dict[str, list[dict[str, Any]]],
    qids: set[str],
    *,
    top_k: int,
    blend_weight: float,
    retriever_name: str,
) -> list[dict[str, Any]]:
    dummy_qrels: dict[str, dict[str, float]] = {}
    matrix, _, meta = build_matrix(features_by_qid, qids, dummy_qrels, top_k)
    model_scores = predict_probabilities(model, matrix)

    by_qid_model: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    by_qid_base: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for score, meta_row in zip(model_scores, meta, strict=False):
        qid = meta_row["qid"]
        item = meta_row["item"]
        by_qid_model.setdefault(qid, []).append((float(score), meta_row))
        by_qid_base.setdefault(qid, []).append((float(item["features"].get("base_rank_score", 0.0)), meta_row))

    norm_model = minmax_by_qid(by_qid_model)
    norm_base = minmax_by_qid(by_qid_base)
    base_lookup = {
        (qid, meta_row["passage_id"]): score
        for qid, rows in norm_base.items()
        for score, meta_row in rows
    }

    predictions: list[dict[str, Any]] = []
    for qid in sorted(norm_model):
        scored = []
        for model_score, meta_row in norm_model[qid]:
            base_score = base_lookup[(qid, meta_row["passage_id"])]
            score = blend_weight * base_score + (1.0 - blend_weight) * model_score
            scored.append((score, meta_row))
        scored.sort(
            key=lambda pair: (
                -pair[0],
                int(pair[1]["item"]["base_rank"]),
                str(pair[1]["passage_id"]),
            )
        )
        for rank, (score, meta_row) in enumerate(scored[:top_k], start=1):
            item = meta_row["item"]
            row = item["row"]
            predictions.append(
                {
                    "question_id": row["question_id"],
                    "passage_id": row["passage_id"],
                    "rank": rank,
                    "score": float(score),
                    "retriever": retriever_name,
                    "metadata": {
                        "base_rank": item["base_rank"],
                        "blend_weight": blend_weight,
                        "model_score_norm": float(1.0 - blend_weight),
                        "features": meta_row["features"],
                        "source_metadata": row.get("metadata", {}),
                    },
                }
            )
    return predictions


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def filter_source_predictions(predictions: list[dict[str, Any]], qids: set[str], top_k: int) -> list[dict[str, Any]]:
    kept = [row for row in predictions if str(row["question_id"]) in qids and int(row["rank"]) <= top_k]
    return kept


def coefficient_table(model: Any) -> list[dict[str, float | str]]:
    if not hasattr(model, "named_steps") or "logisticregression" not in model.named_steps:
        return []
    coef = model.named_steps["logisticregression"].coef_[0]
    rows = [{"feature": name, "coefficient": float(value)} for name, value in zip(FEATURE_NAMES, coef, strict=False)]
    return sorted(rows, key=lambda row: abs(float(row["coefficient"])), reverse=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    paths = config["paths"]

    qrels_path = args.qrels or paths.get("qrels", "data/processed/bioasq_qrels.jsonl")
    question_entities_path = args.question_entities or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    passage_entities_path = args.passage_entities or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")
    question_mesh_path = args.question_mesh or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")
    passage_mesh_path = args.passage_mesh or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")
    relations_path = args.relations or paths.get("primekg_relations", "data/external_knowledge/primekg_project_relations.jsonl")

    predictions = read_jsonl(args.predictions)
    qrels = read_jsonl(qrels_path)
    qrels_by_qid = group_qrels(qrels)
    question_entities = entity_map(read_jsonl(question_entities_path), "question_id")
    passage_entities = entity_map(read_jsonl(passage_entities_path), "passage_id")
    question_mesh = mesh_map(read_jsonl(question_mesh_path), "question_id") if Path(question_mesh_path).exists() else {}
    passage_mesh = mesh_map(read_jsonl(passage_mesh_path), "passage_id") if Path(passage_mesh_path).exists() else {}
    entity_relations = relations_map(read_jsonl(relations_path)) if Path(relations_path).exists() else {}

    features_by_qid = build_feature_rows(
        predictions,
        question_entities,
        passage_entities,
        question_mesh,
        passage_mesh,
        entity_relations,
        structure="knowledge_hypergraph",
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        iterations=args.iterations,
        damping=args.damping,
        max_passage_entities=args.max_passage_entities,
        max_passage_mesh=args.max_passage_mesh,
    )
    all_qids = sorted(features_by_qid)
    splits = split_qids(
        all_qids,
        modulo=args.split_modulo,
        validation_remainders=set(args.validation_remainders),
        test_remainders=set(args.test_remainders),
    )
    train_x, train_y, _ = build_matrix(features_by_qid, splits["train"], qrels_by_qid, args.top_k)
    val_qrels = filter_qrels(qrels, splits["validation"])
    test_qrels = filter_qrels(qrels, splits["test"])

    if train_y.sum() == 0:
        raise ValueError("No positive labels in the training split.")

    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    for c_value in args.c_grid:
        model = make_model(args.model, c_value, seed)
        model.fit(train_x, train_y)
        train_auc = roc_auc_score(train_y, predict_probabilities(model, train_x))
        for blend_weight in args.blend_grid:
            val_predictions = rerank_split(
                model,
                features_by_qid,
                splits["validation"],
                top_k=args.top_k,
                blend_weight=blend_weight,
                retriever_name=f"learning_{args.model}_validation",
            )
            val_metrics = evaluate_retrieval(val_qrels, val_predictions, sorted(set(args.ks)))
            trial = {
                "model": args.model,
                "c_value": float(c_value),
                "blend_weight": float(blend_weight),
                "train_auc": float(train_auc),
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
                best = {"key": key, "trial": trial, "model": model}

    assert best is not None
    selected = best["trial"]
    final_model = make_model(args.model, float(selected["c_value"]), seed)
    final_train_qids = splits["train"] | splits["validation"]
    final_x, final_y, _ = build_matrix(features_by_qid, final_train_qids, qrels_by_qid, args.top_k)
    final_model.fit(final_x, final_y)

    test_predictions = rerank_split(
        final_model,
        features_by_qid,
        splits["test"],
        top_k=args.top_k,
        blend_weight=float(selected["blend_weight"]),
        retriever_name=f"learning_{args.model}",
    )
    write_jsonl(args.output, test_predictions)

    test_metrics = evaluate_retrieval(test_qrels, test_predictions, sorted(set(args.ks)))
    source_test_predictions = filter_source_predictions(predictions, splits["test"], args.top_k)
    source_test_metrics = evaluate_retrieval(test_qrels, source_test_predictions, sorted(set(args.ks)))

    metrics = {
        **test_metrics,
        "timestamp": datetime.now(UTC).isoformat(),
        "qrels": qrels_path,
        "predictions": args.output,
        "source_predictions": args.predictions,
        "source_test_metrics": source_test_metrics,
        "model": args.model,
        "feature_names": FEATURE_NAMES,
        "selected": selected,
        "top_trials": sorted(
            trials,
            key=lambda row: (-row["validation_mrr@10"], -row["validation_recall@10"], -row["validation_ndcg@10"]),
        )[:20],
        "coefficients": coefficient_table(final_model),
        "split": {
            "modulo": args.split_modulo,
            "train_qids": len(splits["train"]),
            "validation_qids": len(splits["validation"]),
            "test_qids": len(splits["test"]),
            "validation_remainders": args.validation_remainders,
            "test_remainders": args.test_remainders,
        },
        "train_rows": int(train_x.shape[0]),
        "train_positive_rows": int(train_y.sum()),
        "final_train_rows": int(final_x.shape[0]),
        "final_train_positive_rows": int(final_y.sum()),
    }
    write_json(args.metrics_output, metrics)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "output": args.output,
        "metrics_output": args.metrics_output,
        "selected": selected,
        "test_mrr@10": test_metrics.get("mrr@10"),
        "source_test_mrr@10": source_test_metrics.get("mrr@10"),
        "test_recall@10": test_metrics.get("recall@10"),
        "source_test_recall@10": source_test_metrics.get("recall@10"),
        "split": metrics["split"],
    }
    write_json(Path(paths["logs_dir"]) / "run_learning_rerank_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
