from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMRanker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import dcg, evaluate_retrieval, group_predictions, group_qrels
from src.knowledge.mesh_hierarchy import (
    hierarchy_feature_values,
    load_mesh_hierarchy,
    shared_mesh_cluster_features,
)
from src.rerank.lambdamart import make_lambdamart_ranker
from src.rerank.hypergraph import build_feature_rows, entity_map, mesh_map, relations_map
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


RETRIEVAL_FEATURES = [
    "base_rank_score",
    "hybrid_score",
    "bm25_score",
    "dense_score",
    "bm25_rank_score",
    "dense_rank_score",
    "rank_percentile",
]
SEMANTIC_FEATURES = [
    "biomedical_semantic_score",
    "biomedical_semantic_rank_score",
]
ENTITY_FEATURES = [
    "entity_overlap_count",
    "entity_jaccard",
    "question_entity_coverage",
    "passage_entity_count",
]
MESH_EXACT_FEATURES = [
    "mesh_overlap_count",
    "mesh_jaccard",
    "question_mesh_coverage",
    "passage_mesh_count",
]
MESH_HIERARCHY_FEATURES = [
    "mesh_hierarchy_exact_count",
    "mesh_parent_match_count",
    "mesh_ancestor_match_count",
    "mesh_sibling_match_count",
    "mesh_tree_similarity_max",
    "mesh_tree_similarity_mean",
    "mesh_tree_distance_min",
    "question_mesh_hierarchy_coverage",
    "passage_mesh_hierarchy_coverage",
    "passage_mesh_specificity",
    "shared_mesh_term_cluster_size",
    "shared_mesh_parent_cluster_size",
    "shared_mesh_term_cluster_ratio",
    "shared_mesh_parent_cluster_ratio",
]
PRIMEKG_FEATURES = [
    "primekg_relation_count",
    "question_relation_coverage",
    "local_primekg_relation_edges",
]
HYPERGRAPH_FEATURES = [
    "hypergraph_score_norm",
    "hypergraph_degree_centrality",
    "hypergraph_degree_centrality_norm",
    "local_num_nodes",
    "local_num_hyperedges",
    "local_shared_entity_edges",
    "local_document_mesh_edges",
    "local_mesh_hierarchy_edges",
    "local_mesh_parent_edges",
    "local_mesh_ancestor_edges",
    "local_primekg_relation_edges",
]
ALL_FEATURES = [
    *RETRIEVAL_FEATURES,
    *SEMANTIC_FEATURES,
    *ENTITY_FEATURES,
    *MESH_EXACT_FEATURES,
    *MESH_HIERARCHY_FEATURES,
    *PRIMEKG_FEATURES,
    *HYPERGRAPH_FEATURES,
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
    parser = argparse.ArgumentParser(description="Run KCH-MedRank LambdaMART reranking and diagnostics.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-label", default="bioasq")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--bm25-predictions", default="outputs/retrieval/bm25_full_top100.jsonl")
    parser.add_argument("--dense-predictions", default="outputs/retrieval/dense_full_top100.jsonl")
    parser.add_argument("--hybrid-predictions", default="outputs/retrieval/hybrid_full_top100.jsonl")
    parser.add_argument("--biomedical-reranker-predictions", default=None)
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--mesh-hierarchy", default="data/external_knowledge/mesh_hierarchy_2026.jsonl")
    parser.add_argument(
        "--enable-mesh-hierarchy-graph-edges",
        action="store_true",
        help="Add MeSH tree/ancestor hyperedges to the local graph. Disabled by default for full CPU runs; hierarchy-aware features remain enabled.",
    )
    parser.add_argument("--relations", default=None)
    parser.add_argument("--output-prefix", default="kch_medrank")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--max-passage-entities", type=int, default=48)
    parser.add_argument("--max-passage-mesh", type=int, default=32)
    parser.add_argument("--split-modulo", type=int, default=5)
    parser.add_argument("--validation-remainders", type=int, nargs="+", default=[3])
    parser.add_argument("--test-remainders", type=int, nargs="+", default=[4])
    parser.add_argument("--max-qids", type=int, default=None)
    parser.add_argument("--num-leaves-grid", type=parse_int_grid, default=parse_int_grid("7,15,31"))
    parser.add_argument("--learning-rate-grid", type=parse_float_grid, default=parse_float_grid("0.03,0.05"))
    parser.add_argument("--n-estimators-grid", type=parse_int_grid, default=parse_int_grid("80,160"))
    parser.add_argument("--blend-grid", type=parse_float_grid, default=parse_float_grid("0,0.1,0.2,0.35"))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def qid_bucket(qid: str, modulo: int) -> int:
    if qid.isdigit():
        return int(qid) % modulo
    return sum(ord(char) for char in qid) % modulo


def split_qids(qids: list[str], modulo: int, validation_remainders: set[int], test_remainders: set[int]) -> dict[str, set[str]]:
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


def minmax_values(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def read_score_predictions(path: str | None) -> dict[tuple[str, str], float]:
    if not path:
        return {}
    scores: dict[tuple[str, str], float] = {}
    for row in read_jsonl(path):
        scores[(str(row["question_id"]), str(row["passage_id"]))] = float(row.get("score", 0.0))
    return scores


def semantic_score(row: dict[str, Any], score_lookup: dict[tuple[str, str], float]) -> float:
    qid = str(row["question_id"])
    pid = str(row["passage_id"])
    if (qid, pid) in score_lookup:
        return score_lookup[(qid, pid)]
    return source_feature(row, "dense_score")


def add_enriched_features(
    features_by_qid: dict[str, list[dict[str, Any]]],
    question_mesh: dict[str, list[dict[str, Any]]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    mesh_hierarchy: dict[str, Any],
    semantic_scores: dict[tuple[str, str], float],
    *,
    top_k: int,
    max_passage_mesh: int,
) -> None:
    for qid, items in features_by_qid.items():
        candidate_mesh = {
            str(item["row"]["passage_id"]): passage_mesh.get(str(item["row"]["passage_id"]), [])
            for item in items
        }
        cluster_features = shared_mesh_cluster_features(
            candidate_mesh,
            mesh_hierarchy,
            max_passage_mesh=max_passage_mesh,
        )
        sem_values = [semantic_score(item["row"], semantic_scores) for item in items]
        sem_rank_scores = minmax_values(sem_values)
        for item, sem_value, sem_rank in zip(items, sem_values, sem_rank_scores, strict=False):
            row = item["row"]
            pid = str(row["passage_id"])
            features = item["features"]
            base_rank = int(item["base_rank"])
            features["hybrid_score"] = float(row.get("score", 0.0))
            features["bm25_score"] = source_feature(row, "bm25_score")
            features["dense_score"] = source_feature(row, "dense_score")
            features["bm25_rank_score"] = source_feature(row, "bm25_rank_score")
            features["dense_rank_score"] = source_feature(row, "dense_rank_score")
            features["rank_percentile"] = 1.0 - ((base_rank - 1) / max(top_k - 1, 1))
            features["biomedical_semantic_score"] = float(sem_value)
            features["biomedical_semantic_rank_score"] = float(sem_rank)
            features.update(
                hierarchy_feature_values(
                    question_mesh.get(qid, []),
                    passage_mesh.get(pid, []),
                    mesh_hierarchy,
                    max_passage_mesh=max_passage_mesh,
                )
            )
            features.update(cluster_features.get(pid, {}))


def build_all_feature_rows(
    predictions: list[dict[str, Any]],
    question_entities: dict[str, list[dict[str, Any]]],
    passage_entities: dict[str, list[dict[str, Any]]],
    question_mesh: dict[str, list[dict[str, Any]]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    mesh_hierarchy: dict[str, Any],
    entity_relations: dict[str, list[dict[str, Any]]],
    semantic_scores: dict[tuple[str, str], float],
    *,
    enable_mesh_hierarchy_graph_edges: bool,
    structure: str,
    top_k: int,
    rrf_k: int,
    iterations: int,
    damping: float,
    max_passage_entities: int,
    max_passage_mesh: int,
) -> dict[str, list[dict[str, Any]]]:
    rows = build_feature_rows(
        predictions,
        question_entities,
        passage_entities,
        question_mesh,
        passage_mesh,
        entity_relations,
        mesh_hierarchy=mesh_hierarchy if enable_mesh_hierarchy_graph_edges else {},
        structure=structure,
        top_k=top_k,
        rrf_k=rrf_k,
        iterations=iterations,
        damping=damping,
        max_passage_entities=max_passage_entities,
        max_passage_mesh=max_passage_mesh,
    )
    add_enriched_features(
        rows,
        question_mesh,
        passage_mesh,
        mesh_hierarchy,
        semantic_scores,
        top_k=top_k,
        max_passage_mesh=max_passage_mesh,
    )
    return rows


def feature_names_for(setting: str) -> list[str]:
    selected = set(ALL_FEATURES)
    if setting == "retrieval_ltr":
        selected = set(RETRIEVAL_FEATURES)
    elif setting == "semantic_no_hypergraph_ltr":
        selected = set(RETRIEVAL_FEATURES + SEMANTIC_FEATURES + ENTITY_FEATURES + MESH_EXACT_FEATURES + MESH_HIERARCHY_FEATURES + PRIMEKG_FEATURES)
    elif setting == "pairwise_graph_ltr":
        selected = set(ALL_FEATURES)
    elif setting == "hypergraph_no_medical_knowledge_ltr":
        selected = set(RETRIEVAL_FEATURES + SEMANTIC_FEATURES + HYPERGRAPH_FEATURES)
    elif setting == "full_kch_medrank":
        selected = set(ALL_FEATURES)
    elif setting == "remove_semantic":
        selected = set(ALL_FEATURES) - set(SEMANTIC_FEATURES)
    elif setting == "remove_mesh_hierarchy":
        selected = set(ALL_FEATURES) - set(MESH_HIERARCHY_FEATURES)
    elif setting == "remove_entity":
        selected = set(ALL_FEATURES) - set(ENTITY_FEATURES)
    elif setting == "remove_hypergraph":
        selected = set(ALL_FEATURES) - set(HYPERGRAPH_FEATURES)
    elif setting == "remove_primekg":
        selected = set(ALL_FEATURES) - set(PRIMEKG_FEATURES)
    else:
        raise ValueError(f"Unsupported KCH-MedRank setting: {setting}")
    return [name for name in ALL_FEATURES if name in selected]


def matrix_for_qids(
    features_by_qid: dict[str, list[dict[str, Any]]],
    qids: set[str],
    qrels_by_qid: dict[str, dict[str, float]],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int], list[dict[str, Any]]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    meta: list[dict[str, Any]] = []
    for qid in sorted(qids):
        q_items = features_by_qid.get(qid, [])
        if not q_items:
            continue
        groups.append(len(q_items))
        gold = qrels_by_qid.get(qid, {})
        for item in q_items:
            pid = str(item["row"]["passage_id"])
            features = item["features"]
            rows.append([float(features.get(name, 0.0)) for name in feature_names])
            labels.append(1 if pid in gold else 0)
            meta.append({"qid": qid, "passage_id": pid, "item": item, "features": dict(features)})
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64), groups, meta


def make_ranker(seed: int, *, num_leaves: int, learning_rate: float, n_estimators: int) -> LGBMRanker:
    return make_lambdamart_ranker(
        seed=seed,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
    )


def grouped_minmax(scored: dict[str, list[tuple[float, dict[str, Any]]]]) -> dict[str, list[tuple[float, dict[str, Any]]]]:
    output: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for qid, rows in scored.items():
        values = [score for score, _ in rows]
        norm = minmax_values(values)
        output[qid] = [(score, meta) for score, (_raw, meta) in zip(norm, rows, strict=False)]
    return output


def rerank_with_model(
    model: LGBMRanker,
    features_by_qid: dict[str, list[dict[str, Any]]],
    qids: set[str],
    qrels_by_qid: dict[str, dict[str, float]],
    feature_names: list[str],
    *,
    blend_weight: float,
    top_k: int,
    retriever_name: str,
) -> list[dict[str, Any]]:
    matrix, _labels, _groups, meta = matrix_for_qids(features_by_qid, qids, qrels_by_qid, feature_names)
    scores = model.predict(matrix) if len(meta) else np.asarray([])
    by_qid_model: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    by_qid_base: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for score, meta_row in zip(scores, meta, strict=False):
        qid = meta_row["qid"]
        by_qid_model[qid].append((float(score), meta_row))
        by_qid_base[qid].append((float(meta_row["item"]["features"].get("base_rank_score", 0.0)), meta_row))
    norm_model = grouped_minmax(by_qid_model)
    norm_base = grouped_minmax(by_qid_base)
    base_lookup = {
        (qid, row["passage_id"]): score
        for qid, rows in norm_base.items()
        for score, row in rows
    }
    predictions: list[dict[str, Any]] = []
    for qid in sorted(norm_model):
        scored = []
        for model_score, meta_row in norm_model[qid]:
            base_score = base_lookup[(qid, meta_row["passage_id"])]
            score = blend_weight * base_score + (1.0 - blend_weight) * model_score
            scored.append((score, meta_row))
        scored.sort(key=lambda pair: (-pair[0], int(pair[1]["item"]["base_rank"]), str(pair[1]["passage_id"])))
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
                        "base_rank": int(item["base_rank"]),
                        "blend_weight": float(blend_weight),
                        "features": meta_row["features"],
                        "source_metadata": row.get("metadata", {}),
                    },
                }
            )
    return predictions


def semantic_only_predictions(
    features_by_qid: dict[str, list[dict[str, Any]]],
    qids: set[str],
    *,
    top_k: int,
    retriever_name: str = "biomedical_semantic_reranker_only",
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for qid in sorted(qids):
        scored = []
        for item in features_by_qid.get(qid, []):
            score = float(item["features"].get("biomedical_semantic_score", 0.0))
            scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], int(pair[1]["base_rank"]), str(pair[1]["row"]["passage_id"])))
        for rank, (score, item) in enumerate(scored[:top_k], start=1):
            row = item["row"]
            predictions.append(
                {
                    "question_id": row["question_id"],
                    "passage_id": row["passage_id"],
                    "rank": rank,
                    "score": float(score),
                    "retriever": retriever_name,
                    "metadata": {
                        "base_rank": int(item["base_rank"]),
                        "features": item["features"],
                        "source_metadata": row.get("metadata", {}),
                    },
                }
            )
    return predictions


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def filter_predictions(predictions: list[dict[str, Any]], qids: set[str], top_k: int) -> list[dict[str, Any]]:
    return [row for row in predictions if str(row["question_id"]) in qids and int(row.get("rank", top_k + 1)) <= top_k]


def add_evidence_coverage(metrics: dict[str, Any], ks: list[int]) -> dict[str, Any]:
    enriched = dict(metrics)
    for k in ks:
        if f"recall@{k}" in enriched:
            enriched[f"evidence_coverage@{k}"] = enriched[f"recall@{k}"]
    return enriched


def per_query_metric(gold: dict[str, float], rows: list[dict[str, Any]], metric: str, k: int) -> float:
    ranked_ids = [str(row["passage_id"]) for row in rows[:k]]
    gold_ids = set(gold)
    hits = gold_ids & set(ranked_ids)
    if metric in {"recall", "evidence_coverage"}:
        return len(hits) / len(gold_ids) if gold_ids else 0.0
    if metric == "mrr":
        for rank, pid in enumerate(ranked_ids, start=1):
            if pid in gold_ids:
                return 1.0 / rank
        return 0.0
    if metric == "ndcg":
        gains = [gold.get(pid, 0.0) for pid in ranked_ids]
        ideal_gains = sorted(gold.values(), reverse=True)[:k]
        ideal = dcg(ideal_gains)
        return dcg(gains) / ideal if ideal > 0 else 0.0
    raise ValueError(metric)


def paired_bootstrap(
    qrels: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    candidate_predictions: list[dict[str, Any]],
    *,
    baseline_label: str,
    candidate_label: str,
    dataset_label: str,
    ks: list[int],
    metrics: list[str],
    num_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    qrels_by_qid = group_qrels(qrels)
    baseline_by_qid = group_predictions(baseline_predictions)
    candidate_by_qid = group_predictions(candidate_predictions)
    qids = sorted(set(qrels_by_qid) & set(baseline_by_qid) & set(candidate_by_qid))
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    for metric in metrics:
        for k in ks:
            baseline_values = np.asarray([per_query_metric(qrels_by_qid[qid], baseline_by_qid.get(qid, []), metric, k) for qid in qids])
            candidate_values = np.asarray([per_query_metric(qrels_by_qid[qid], candidate_by_qid.get(qid, []), metric, k) for qid in qids])
            delta = candidate_values - baseline_values
            boot = np.empty(num_bootstrap, dtype=np.float64)
            n = len(delta)
            for idx in range(num_bootstrap):
                sample_idx = rng.integers(0, n, size=n)
                boot[idx] = float(np.mean(delta[sample_idx]))
            p_lower = (float(np.sum(boot <= 0.0)) + 1.0) / (num_bootstrap + 1.0)
            p_upper = (float(np.sum(boot >= 0.0)) + 1.0) / (num_bootstrap + 1.0)
            baseline_mean = float(np.mean(baseline_values))
            observed = float(np.mean(delta))
            results.append(
                {
                    "metric": metric,
                    "k": k,
                    "baseline_mean": baseline_mean,
                    "candidate_mean": float(np.mean(candidate_values)),
                    "delta": observed,
                    "relative_delta_percent": observed / baseline_mean * 100.0 if baseline_mean else 0.0,
                    "ci_lower": float(np.quantile(boot, 0.025)),
                    "ci_upper": float(np.quantile(boot, 0.975)),
                    "p_value_two_sided": min(1.0, 2.0 * min(p_lower, p_upper)),
                }
            )
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": dataset_label,
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "num_paired_questions": len(qids),
        "num_bootstrap": num_bootstrap,
        "seed": seed,
        "results": results,
    }


def hard_subset_qids(qrels: list[dict[str, Any]], hybrid_predictions: list[dict[str, Any]], qids: set[str]) -> set[str]:
    qrels_by_qid = group_qrels(qrels)
    preds_by_qid = group_predictions(hybrid_predictions)
    hard: set[str] = set()
    for qid in qids:
        gold = set(qrels_by_qid.get(qid, {}))
        ranked = preds_by_qid.get(qid, [])
        top10 = {str(row["passage_id"]) for row in ranked[:10]}
        top100 = {str(row["passage_id"]) for row in ranked[:100]}
        if gold & top100 and not (gold & top10):
            hard.add(qid)
    return hard


def train_setting(
    label: str,
    features_by_qid: dict[str, list[dict[str, Any]]],
    feature_names: list[str],
    splits: dict[str, set[str]],
    qrels_by_qid: dict[str, dict[str, float]],
    validation_qrels: list[dict[str, Any]],
    args: argparse.Namespace,
    seed: int,
) -> tuple[LGBMRanker, dict[str, Any]]:
    train_x, train_y, train_group, _ = matrix_for_qids(features_by_qid, splits["train"], qrels_by_qid, feature_names)
    if int(train_y.sum()) == 0:
        raise ValueError(f"No positive labels in train split for {label}.")
    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    for num_leaves in args.num_leaves_grid:
        for learning_rate in args.learning_rate_grid:
            for n_estimators in args.n_estimators_grid:
                model = make_ranker(seed, num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators)
                model.fit(train_x, train_y, group=train_group)
                for blend_weight in args.blend_grid:
                    val_predictions = rerank_with_model(
                        model,
                        features_by_qid,
                        splits["validation"],
                        qrels_by_qid,
                        feature_names,
                        blend_weight=blend_weight,
                        top_k=args.top_k,
                        retriever_name=f"{label}_validation",
                    )
                    val_metrics = evaluate_retrieval(validation_qrels, val_predictions, sorted(set(args.ks)))
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
                    key = (trial["validation_mrr@10"], trial["validation_recall@10"], trial["validation_ndcg@10"], -trial["blend_weight"])
                    if best is None or key > best["key"]:
                        best = {"key": key, "trial": trial}
    assert best is not None
    selected = best["trial"]
    final_qids = splits["train"] | splits["validation"]
    final_x, final_y, final_group, _ = matrix_for_qids(features_by_qid, final_qids, qrels_by_qid, feature_names)
    final_model = make_ranker(
        seed,
        num_leaves=int(selected["num_leaves"]),
        learning_rate=float(selected["learning_rate"]),
        n_estimators=int(selected["n_estimators"]),
    )
    final_model.fit(final_x, final_y, group=final_group)
    diagnostics = {
        "label": label,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "selected": selected,
        "top_trials": sorted(trials, key=lambda row: (-row["validation_mrr@10"], -row["validation_recall@10"], -row["validation_ndcg@10"]))[:10],
        "train_rows": int(train_x.shape[0]),
        "train_positive_rows": int(train_y.sum()),
        "final_train_rows": int(final_x.shape[0]),
        "final_train_positive_rows": int(final_y.sum()),
    }
    importances = final_model.feature_importances_
    diagnostics["feature_importance"] = sorted(
        [
            {"feature": name, "importance": float(value)}
            for name, value in zip(feature_names, importances, strict=False)
        ],
        key=lambda row: float(row["importance"]),
        reverse=True,
    )
    return final_model, diagnostics


def write_table(path_csv: str | Path, path_md: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    Path(path_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(path_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    Path(path_md).parent.mkdir(parents=True, exist_ok=True)
    Path(path_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_row(method: str, metrics: dict[str, Any], source_metrics: dict[str, Any] | None = None) -> dict[str, str]:
    row = {"method": method}
    for key in ["recall@5", "recall@10", "mrr@10", "ndcg@10", "evidence_coverage@10", "recall@100", "mrr@100", "ndcg@100"]:
        row[key] = f"{float(metrics.get(key, 0.0)):.4f}" if key in metrics else ""
    if source_metrics:
        row["delta_mrr@10"] = f"{float(metrics.get('mrr@10', 0.0)) - float(source_metrics.get('mrr@10', 0.0)):+.4f}"
        row["delta_recall@10"] = f"{float(metrics.get('recall@10', 0.0)) - float(source_metrics.get('recall@10', 0.0)):+.4f}"
    else:
        row["delta_mrr@10"] = ""
        row["delta_recall@10"] = ""
    return row


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(seed)
    paths = config["paths"]

    qrels_path = args.qrels or paths.get("qrels", "data/processed/bioasq_qrels.jsonl")
    question_entities_path = args.question_entities or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    passage_entities_path = args.passage_entities or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")
    question_mesh_path = args.question_mesh or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")
    passage_mesh_path = args.passage_mesh or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")
    relations_path = args.relations or paths.get("primekg_relations", "data/external_knowledge/primekg_project_relations.jsonl")

    bm25_predictions = read_jsonl(args.bm25_predictions)
    dense_predictions = read_jsonl(args.dense_predictions)
    hybrid_predictions = read_jsonl(args.hybrid_predictions)
    if args.max_qids is not None:
        selected_qids = set(sorted({str(row["question_id"]) for row in hybrid_predictions})[: args.max_qids])
        bm25_predictions = [row for row in bm25_predictions if str(row["question_id"]) in selected_qids]
        dense_predictions = [row for row in dense_predictions if str(row["question_id"]) in selected_qids]
        hybrid_predictions = [row for row in hybrid_predictions if str(row["question_id"]) in selected_qids]

    qrels = read_jsonl(qrels_path)
    qrels_by_qid = group_qrels(qrels)
    question_entities = entity_map(read_jsonl(question_entities_path), "question_id")
    passage_entities = entity_map(read_jsonl(passage_entities_path), "passage_id")
    question_mesh = mesh_map(read_jsonl(question_mesh_path), "question_id") if Path(question_mesh_path).exists() else {}
    passage_mesh = mesh_map(read_jsonl(passage_mesh_path), "passage_id") if Path(passage_mesh_path).exists() else {}
    mesh_hierarchy = load_mesh_hierarchy(read_jsonl(args.mesh_hierarchy)) if Path(args.mesh_hierarchy).exists() else {}
    entity_relations = relations_map(read_jsonl(relations_path)) if Path(relations_path).exists() else {}
    semantic_scores = read_score_predictions(args.biomedical_reranker_predictions)

    semantic_source = args.biomedical_reranker_predictions or "hybrid_metadata.dense_score_fallback"
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Building full knowledge-hypergraph features...", flush=True)
    full_features = build_all_feature_rows(
        hybrid_predictions,
        question_entities,
        passage_entities,
        question_mesh,
        passage_mesh,
        mesh_hierarchy,
        entity_relations,
        semantic_scores,
        enable_mesh_hierarchy_graph_edges=args.enable_mesh_hierarchy_graph_edges,
        structure="knowledge_hypergraph",
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        iterations=args.iterations,
        damping=args.damping,
        max_passage_entities=args.max_passage_entities,
        max_passage_mesh=args.max_passage_mesh,
    )
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Building pairwise graph features...", flush=True)
    pairwise_features = build_all_feature_rows(
        hybrid_predictions,
        question_entities,
        passage_entities,
        question_mesh,
        passage_mesh,
        mesh_hierarchy,
        entity_relations,
        semantic_scores,
        enable_mesh_hierarchy_graph_edges=args.enable_mesh_hierarchy_graph_edges,
        structure="pairwise_graph",
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        iterations=args.iterations,
        damping=args.damping,
        max_passage_entities=args.max_passage_entities,
        max_passage_mesh=args.max_passage_mesh,
    )
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Building no-knowledge hypergraph features...", flush=True)
    no_knowledge_features = build_all_feature_rows(
        hybrid_predictions,
        question_entities,
        passage_entities,
        question_mesh,
        passage_mesh,
        mesh_hierarchy,
        entity_relations,
        semantic_scores,
        enable_mesh_hierarchy_graph_edges=args.enable_mesh_hierarchy_graph_edges,
        structure="no_knowledge_hypergraph",
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        iterations=args.iterations,
        damping=args.damping,
        max_passage_entities=args.max_passage_entities,
        max_passage_mesh=args.max_passage_mesh,
    )
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Feature construction complete.", flush=True)

    all_qids = sorted(full_features)
    splits = split_qids(
        all_qids,
        args.split_modulo,
        set(args.validation_remainders),
        set(args.test_remainders),
    )
    validation_qrels = filter_qrels(qrels, splits["validation"])
    test_qrels = filter_qrels(qrels, splits["test"])
    hard_qids = hard_subset_qids(qrels, hybrid_predictions, splits["test"])
    hard_qrels = filter_qrels(qrels, hard_qids)

    output_dir = Path("outputs/rerank")
    metrics_dir = Path("results/metrics")
    tables_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    baseline_predictions = {
        "BM25": filter_predictions(bm25_predictions, splits["test"], args.top_k),
        "Dense": filter_predictions(dense_predictions, splits["test"], args.top_k),
        "Hybrid RRF": filter_predictions(hybrid_predictions, splits["test"], args.top_k),
    }
    baseline_metrics = {
        name: add_evidence_coverage(evaluate_retrieval(test_qrels, rows, sorted(set(args.ks))), sorted(set(args.ks)))
        for name, rows in baseline_predictions.items()
    }

    semantic_predictions = semantic_only_predictions(full_features, splits["test"], top_k=args.top_k)
    semantic_output = output_dir / f"{args.output_prefix}_semantic_only_test_top{args.top_k}.jsonl"
    write_jsonl(semantic_output, semantic_predictions)
    semantic_metrics = add_evidence_coverage(evaluate_retrieval(test_qrels, semantic_predictions, sorted(set(args.ks))), sorted(set(args.ks)))

    settings = [
        ("Retrieval-feature-only LambdaMART", "retrieval_ltr", full_features),
        ("LambdaMART + biomedical semantic without hypergraph", "semantic_no_hypergraph_ltr", full_features),
        ("Pairwise graph LTR", "pairwise_graph_ltr", pairwise_features),
        ("Hypergraph LTR without medical knowledge", "hypergraph_no_medical_knowledge_ltr", no_knowledge_features),
        ("Full KCH-MedRank", "full_kch_medrank", full_features),
        ("Remove biomedical semantic reranker", "remove_semantic", full_features),
        ("Remove MeSH hierarchy features", "remove_mesh_hierarchy", full_features),
        ("Remove biomedical entity features", "remove_entity", full_features),
        ("Remove hypergraph diffusion and centrality", "remove_hypergraph", full_features),
        ("Remove PrimeKG relation features", "remove_primekg", full_features),
    ]
    setting_metrics: dict[str, dict[str, Any]] = {}
    setting_predictions: dict[str, list[dict[str, Any]]] = {}
    setting_diagnostics: dict[str, Any] = {}

    for display_name, setting_name, feature_rows in settings:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Training {setting_name}...", flush=True)
        feature_names = feature_names_for(setting_name)
        model, diagnostics = train_setting(
            setting_name,
            feature_rows,
            feature_names,
            splits,
            qrels_by_qid,
            validation_qrels,
            args,
            seed,
        )
        predictions = rerank_with_model(
            model,
            feature_rows,
            splits["test"],
            qrels_by_qid,
            feature_names,
            blend_weight=float(diagnostics["selected"]["blend_weight"]),
            top_k=args.top_k,
            retriever_name=setting_name,
        )
        output_path = output_dir / f"{args.output_prefix}_{setting_name}_test_top{args.top_k}.jsonl"
        write_jsonl(output_path, predictions)
        metrics = add_evidence_coverage(evaluate_retrieval(test_qrels, predictions, sorted(set(args.ks))), sorted(set(args.ks)))
        hard_metrics = add_evidence_coverage(evaluate_retrieval(hard_qrels, filter_predictions(predictions, hard_qids, args.top_k), sorted(set(args.ks))), sorted(set(args.ks)))
        metrics["hard_subset_metrics"] = hard_metrics
        metrics["predictions"] = str(output_path)
        metrics["display_name"] = display_name
        metrics["setting_name"] = setting_name
        setting_metrics[display_name] = metrics
        setting_predictions[display_name] = predictions
        setting_diagnostics[setting_name] = diagnostics
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Finished {setting_name}.", flush=True)

    full_predictions = setting_predictions["Full KCH-MedRank"]
    bootstrap_vs_hybrid = paired_bootstrap(
        test_qrels,
        baseline_predictions["Hybrid RRF"],
        full_predictions,
        baseline_label="Hybrid RRF",
        candidate_label="Full KCH-MedRank",
        dataset_label=args.dataset_label,
        ks=[10],
        metrics=["mrr", "recall", "ndcg", "evidence_coverage"],
        num_bootstrap=args.bootstrap_samples,
        seed=seed,
    )
    bootstrap_vs_semantic = paired_bootstrap(
        test_qrels,
        semantic_predictions,
        full_predictions,
        baseline_label="Biomedical semantic reranker only",
        candidate_label="Full KCH-MedRank",
        dataset_label=args.dataset_label,
        ks=[10],
        metrics=["mrr", "recall", "ndcg", "evidence_coverage"],
        num_bootstrap=args.bootstrap_samples,
        seed=seed + 171,
    )
    write_json(metrics_dir / f"{args.output_prefix}_bootstrap_vs_hybrid.json", bootstrap_vs_hybrid)
    write_json(metrics_dir / f"{args.output_prefix}_bootstrap_vs_semantic.json", bootstrap_vs_semantic)

    full_metrics_payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset_label": args.dataset_label,
        "qrels": qrels_path,
        "hybrid_predictions": args.hybrid_predictions,
        "semantic_source": semantic_source,
        "mesh_hierarchy": args.mesh_hierarchy,
        "enable_mesh_hierarchy_graph_edges": args.enable_mesh_hierarchy_graph_edges,
        "top_k": args.top_k,
        "split": {
            "modulo": args.split_modulo,
            "train_qids": len(splits["train"]),
            "validation_qids": len(splits["validation"]),
            "test_qids": len(splits["test"]),
            "hard_subset_qids": len(hard_qids),
            "validation_remainders": args.validation_remainders,
            "test_remainders": args.test_remainders,
            "max_qids": args.max_qids,
        },
        "baseline_metrics": baseline_metrics,
        "biomedical_semantic_reranker_only": semantic_metrics,
        "setting_metrics": setting_metrics,
        "diagnostics": setting_diagnostics,
        "bootstrap_vs_hybrid": bootstrap_vs_hybrid,
        "bootstrap_vs_semantic": bootstrap_vs_semantic,
    }
    write_json(metrics_dir / f"{args.output_prefix}_metrics.json", full_metrics_payload)

    write_json(metrics_dir / f"{args.output_prefix}_full_kch_medrank_metrics.json", setting_metrics["Full KCH-MedRank"])
    write_json(metrics_dir / f"{args.output_prefix}_semantic_only_metrics.json", semantic_metrics)

    rows = []
    hybrid_metrics = baseline_metrics["Hybrid RRF"]
    for name in ["BM25", "Dense", "Hybrid RRF"]:
        rows.append(table_row(name, baseline_metrics[name], hybrid_metrics if name != "Hybrid RRF" else None))
    rows.append(table_row("Biomedical semantic reranker only", semantic_metrics, hybrid_metrics))
    for name, metrics in setting_metrics.items():
        rows.append(table_row(name, metrics, hybrid_metrics))
    columns = [
        "method",
        "recall@5",
        "recall@10",
        "mrr@10",
        "ndcg@10",
        "evidence_coverage@10",
        "recall@100",
        "mrr@100",
        "ndcg@100",
        "delta_mrr@10",
        "delta_recall@10",
    ]
    write_table(tables_dir / f"{args.output_prefix}_retrieval.csv", tables_dir / f"{args.output_prefix}_retrieval.md", rows, columns)

    hard_rows = []
    for name in ["BM25", "Dense", "Hybrid RRF"]:
        hard_pred = filter_predictions(baseline_predictions[name], hard_qids, args.top_k)
        hard_rows.append(table_row(name, add_evidence_coverage(evaluate_retrieval(hard_qrels, hard_pred, sorted(set(args.ks))), sorted(set(args.ks)))))
    hard_rows.append(table_row("Biomedical semantic reranker only", add_evidence_coverage(evaluate_retrieval(hard_qrels, filter_predictions(semantic_predictions, hard_qids, args.top_k), sorted(set(args.ks))), sorted(set(args.ks)))))
    for name, metrics in setting_metrics.items():
        hard_rows.append(table_row(name, metrics.get("hard_subset_metrics", {})))
    write_table(tables_dir / f"{args.output_prefix}_hard_subset.csv", tables_dir / f"{args.output_prefix}_hard_subset.md", hard_rows, columns)

    importance_rows = [
        {
            "feature": row["feature"],
            "importance": f"{float(row['importance']):.4f}",
        }
        for row in setting_diagnostics["full_kch_medrank"].get("feature_importance", [])[:30]
    ]
    write_table(
        tables_dir / f"{args.output_prefix}_feature_importance.csv",
        tables_dir / f"{args.output_prefix}_feature_importance.md",
        importance_rows,
        ["feature", "importance"],
    )

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "metrics": str(metrics_dir / f"{args.output_prefix}_metrics.json"),
        "table": str(tables_dir / f"{args.output_prefix}_retrieval.md"),
        "hard_subset_table": str(tables_dir / f"{args.output_prefix}_hard_subset.md"),
        "semantic_source": semantic_source,
        "hybrid_mrr@10": hybrid_metrics.get("mrr@10"),
        "semantic_mrr@10": semantic_metrics.get("mrr@10"),
        "full_mrr@10": setting_metrics["Full KCH-MedRank"].get("mrr@10"),
        "hybrid_recall@10": hybrid_metrics.get("recall@10"),
        "full_recall@10": setting_metrics["Full KCH-MedRank"].get("recall@10"),
        "hard_subset_qids": len(hard_qids),
    }
    write_json(Path(paths.get("logs_dir", "logs")) / f"run_{args.output_prefix}_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
