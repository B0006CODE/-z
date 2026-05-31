from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from lightgbm import LGBMRanker

from src.evaluation.retrieval_metrics import evaluate_retrieval, group_qrels
from src.hypergraph.counterfactual import COUNTERFACTUAL_FEATURES
from src.rerank.lambdamart import make_lambdamart_ranker
from src.rerank.selective_gate import GateRule, intervention_for_query
from src.rerank.slate_optimizer import SlateWeights, optimize_slate


RETRIEVAL_FEATURES = [
    "base_rank_score",
    "hybrid_score",
    "bm25_score",
    "dense_score",
    "rank_percentile",
]

SEMANTIC_FEATURES = [
    "biomedical_semantic_score",
    "semantic_score_norm",
    "medcpt_score",
]

FLAT_MEDICAL_FEATURES = [
    "direct_entity_overlap_count",
    "direct_mesh_overlap_count",
    "relation_entity_overlap_count",
    "entity_jaccard",
    "question_entity_coverage",
    "mesh_jaccard",
    "question_mesh_coverage",
    "passage_mesh_specificity",
    "mesh_hierarchy_exact_count",
    "mesh_parent_match_count",
    "mesh_ancestor_match_count",
    "mesh_sibling_match_count",
    "mesh_tree_similarity_max",
    "mesh_tree_similarity_mean",
    "question_mesh_hierarchy_coverage",
]

PRECISION_HYPERGRAPH_FEATURES = [
    "precision_hypergraph_score",
    "precision_hypergraph_raw_score",
    "rare_entity_bridge_score",
    "mesh_specificity_score",
    "mesh_hierarchy_bridge_score",
    "shared_seed_support_score",
    "semantic_graph_agreement",
    "synonym_cluster_score",
    "relation_tuple_score",
    "local_hyperedge_count",
    "query_anchor_count",
]

PENALTY_FEATURES = [
    "broad_concept_penalty",
    "hierarchy_only_penalty",
    "high_df_concept_penalty",
    "excessive_shared_cluster_penalty",
    "semantic_graph_disagreement_penalty",
]

ALL_V4_FEATURES = [
    *RETRIEVAL_FEATURES,
    *SEMANTIC_FEATURES,
    *FLAT_MEDICAL_FEATURES,
    *PRECISION_HYPERGRAPH_FEATURES,
    *PENALTY_FEATURES,
    *COUNTERFACTUAL_FEATURES,
]


def feature_names_for(setting: str) -> list[str]:
    selected = set(ALL_V4_FEATURES)
    if setting == "retrieval_ltr":
        selected = set(RETRIEVAL_FEATURES)
    elif setting == "flat_biomedical_ltr":
        selected = set(RETRIEVAL_FEATURES + SEMANTIC_FEATURES + FLAT_MEDICAL_FEATURES)
    elif setting == "pairwise_graph_ltr":
        selected = set(RETRIEVAL_FEATURES + SEMANTIC_FEATURES + FLAT_MEDICAL_FEATURES + PENALTY_FEATURES)
    elif setting == "full_kch_v4":
        selected = set(ALL_V4_FEATURES)
    elif setting == "without_counterfactual":
        selected = set(ALL_V4_FEATURES) - set(COUNTERFACTUAL_FEATURES)
    elif setting == "without_mesh_hierarchy":
        selected = set(ALL_V4_FEATURES) - {
            "mesh_hierarchy_bridge_score",
            "cf_gain_mesh_hierarchy",
            "score_without_mesh_hierarchy_edges",
            "mesh_parent_match_count",
            "mesh_ancestor_match_count",
            "mesh_sibling_match_count",
            "mesh_tree_similarity_max",
            "mesh_tree_similarity_mean",
            "question_mesh_hierarchy_coverage",
        }
    elif setting == "without_rare_entity":
        selected = set(ALL_V4_FEATURES) - {
            "rare_entity_bridge_score",
            "cf_gain_rare_entity",
            "score_without_rare_entity_edges",
            "direct_entity_overlap_count",
            "entity_jaccard",
            "question_entity_coverage",
        }
    elif setting == "without_shared_seed":
        selected = set(ALL_V4_FEATURES) - {
            "shared_seed_support_score",
            "cf_gain_shared_seed",
            "score_without_shared_seed_edges",
        }
    elif setting == "without_semantic_graph_agreement":
        selected = set(ALL_V4_FEATURES) - {
            "semantic_graph_agreement",
            "semantic_graph_disagreement_penalty",
            "cf_gain_semantic_agreement",
            "score_without_semantic_agreement",
        }
    elif setting == "without_broad_high_df_penalties":
        selected = set(ALL_V4_FEATURES) - set(PENALTY_FEATURES)
    elif setting == "without_primekg_relation":
        selected = set(ALL_V4_FEATURES) - {
            "relation_tuple_score",
            "relation_entity_overlap_count",
            "cf_gain_relation",
            "score_without_relation_edges",
        }
    else:
        raise ValueError(f"Unsupported KCH v4 setting: {setting}")
    return [name for name in ALL_V4_FEATURES if name in selected]


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
        items = features_by_qid.get(qid, [])
        if not items:
            continue
        groups.append(len(items))
        gold = qrels_by_qid.get(qid, {})
        for item in items:
            pid = str(item["row"]["passage_id"])
            features = item["features"]
            rows.append([float(features.get(name, 0.0)) for name in feature_names])
            labels.append(1 if pid in gold else 0)
            meta.append({"qid": qid, "pid": pid, "item": item})
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64), groups, meta


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def score_items(
    model: LGBMRanker,
    features_by_qid: dict[str, list[dict[str, Any]]],
    qids: set[str],
    qrels_by_qid: dict[str, dict[str, float]],
    feature_names: list[str],
) -> dict[str, list[dict[str, Any]]]:
    matrix, _labels, _groups, meta = matrix_for_qids(features_by_qid, qids, qrels_by_qid, feature_names)
    scores = model.predict(matrix) if len(meta) else np.asarray([])
    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score, meta_row in zip(scores, meta, strict=False):
        item = meta_row["item"]
        by_qid[meta_row["qid"]].append(
            {
                "qid": meta_row["qid"],
                "pid": meta_row["pid"],
                "item": item,
                "raw_model_score": float(score),
                "base_score": float(item["features"].get("base_rank_score", 0.0)),
                "features": dict(item["features"]),
                "anchor_concepts": list(item.get("anchor_concepts", [])),
                "base_rank": int(item.get("base_rank", 999999)),
            }
        )
    for rows in by_qid.values():
        model_norm = _minmax([row["raw_model_score"] for row in rows])
        base_norm = _minmax([row["base_score"] for row in rows])
        for row, model_value, base_value in zip(rows, model_norm, base_norm, strict=False):
            row["model_score_norm"] = float(model_value)
            row["base_score_norm"] = float(base_value)
    return dict(by_qid)


def build_predictions(
    scored_by_qid: dict[str, list[dict[str, Any]]],
    *,
    retriever_name: str,
    top_k: int,
    blend_weight: float,
    gate_rule: GateRule | None = None,
    fixed_intervention_strength: float = 1.0,
    slate_weights: SlateWeights | None = None,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    slate_weights = slate_weights or SlateWeights(enabled=False)
    for qid in sorted(scored_by_qid):
        items = scored_by_qid[qid]
        gate_items = [{"features": row["features"], "base_rank": row["base_rank"]} for row in items]
        if gate_rule is not None:
            intervention_strength, gate_reason, gate_stats = intervention_for_query(gate_items, gate_rule)
        else:
            intervention_strength = fixed_intervention_strength
            gate_reason = "fixed_intervention"
            gate_stats = {}
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            model_combo = blend_weight * item["base_score_norm"] + (1.0 - blend_weight) * item["model_score_norm"]
            final_score = (1.0 - intervention_strength) * item["base_score_norm"] + intervention_strength * model_combo
            scored.append((float(final_score), item))
        scored.sort(key=lambda pair: (-pair[0], int(pair[1]["base_rank"]), str(pair[1]["pid"])))
        effective_slate_weights = slate_weights if intervention_strength > 0.0 else SlateWeights(enabled=False)
        slate_top = optimize_slate(scored, weights=effective_slate_weights, slate_k=min(10, top_k))
        selected_keys = {(row["qid"], row["pid"]) for _score, row in slate_top}
        tail = [pair for pair in scored if (pair[1]["qid"], pair[1]["pid"]) not in selected_keys]
        ranked = [*slate_top, *tail][:top_k]
        for rank, (score, scored_item) in enumerate(ranked, start=1):
            row = scored_item["item"]["row"]
            predictions.append(
                {
                    "question_id": row["question_id"],
                    "passage_id": row["passage_id"],
                    "rank": rank,
                    "score": float(score),
                    "retriever": retriever_name,
                    "metadata": {
                        "base_rank": int(scored_item["base_rank"]),
                        "model_score": float(scored_item["raw_model_score"]),
                        "model_score_norm": float(scored_item["model_score_norm"]),
                        "base_score_norm": float(scored_item["base_score_norm"]),
                        "blend_weight": float(blend_weight),
                        "intervention_strength": float(intervention_strength),
                        "gate_reason": gate_reason,
                        "final_blend_weight": float(intervention_strength * (1.0 - blend_weight)),
                        "gate_stats": gate_stats,
                        "slate_weights": effective_slate_weights.__dict__,
                        "anchor_concepts": scored_item.get("anchor_concepts", []),
                        "features": scored_item["features"],
                        "source_metadata": row.get("metadata", {}),
                    },
                }
            )
    return predictions


def train_lambdamart(
    setting: str,
    features_by_qid: dict[str, list[dict[str, Any]]],
    train_qids: set[str],
    validation_qids: set[str],
    qrels: list[dict[str, Any]],
    *,
    seed: int,
    num_leaves_grid: list[int],
    learning_rate_grid: list[float],
    n_estimators_grid: list[int],
    blend_grid: list[float],
    ks: list[int],
) -> tuple[LGBMRanker, LGBMRanker, list[str], dict[str, Any]]:
    qrels_by_qid = group_qrels(qrels)
    feature_names = feature_names_for(setting)
    train_x, train_y, train_group, _ = matrix_for_qids(features_by_qid, train_qids, qrels_by_qid, feature_names)
    if int(train_y.sum()) == 0:
        raise ValueError(f"No positive labels in train split for {setting}.")
    validation_qrels = [row for row in qrels if str(row["question_id"]) in validation_qids]
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_model: LGBMRanker | None = None
    for num_leaves in num_leaves_grid:
        for learning_rate in learning_rate_grid:
            for n_estimators in n_estimators_grid:
                model = make_lambdamart_ranker(
                    seed=seed,
                    num_leaves=num_leaves,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                )
                model.fit(train_x, train_y, group=train_group)
                scored = score_items(model, features_by_qid, validation_qids, qrels_by_qid, feature_names)
                for blend_weight in blend_grid:
                    predictions = build_predictions(
                        scored,
                        retriever_name=f"{setting}_validation",
                        top_k=max(ks),
                        blend_weight=blend_weight,
                        gate_rule=None,
                        fixed_intervention_strength=1.0,
                    )
                    metrics = evaluate_retrieval(validation_qrels, predictions, ks)
                    trial = {
                        "num_leaves": int(num_leaves),
                        "learning_rate": float(learning_rate),
                        "n_estimators": int(n_estimators),
                        "blend_weight": float(blend_weight),
                        "validation_ndcg@10": float(metrics.get("ndcg@10", 0.0)),
                        "validation_mrr@10": float(metrics.get("mrr@10", 0.0)),
                        "validation_recall@10": float(metrics.get("recall@10", 0.0)),
                        "validation_map@10": float(metrics.get("map@10", 0.0)),
                        "validation_hit@10": float(metrics.get("hit@10", 0.0)),
                    }
                    key = (
                        trial["validation_ndcg@10"],
                        trial["validation_mrr@10"],
                        trial["validation_recall@10"],
                        trial["validation_map@10"],
                        -blend_weight,
                    )
                    trial["selection_key"] = [float(value) for value in key]
                    trials.append(trial)
                    if best is None or key > best["key"]:
                        best = {"key": key, "trial": trial}
                        best_model = model
    if best is None or best_model is None:
        raise ValueError(f"No validation trial completed for {setting}.")

    selected = best["trial"]
    final_qids = set(train_qids) | set(validation_qids)
    final_x, final_y, final_group, _ = matrix_for_qids(features_by_qid, final_qids, qrels_by_qid, feature_names)
    final_model = make_lambdamart_ranker(
        seed=seed,
        num_leaves=int(selected["num_leaves"]),
        learning_rate=float(selected["learning_rate"]),
        n_estimators=int(selected["n_estimators"]),
    )
    final_model.fit(final_x, final_y, group=final_group)
    diagnostics = {
        "setting": setting,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "selected": selected,
        "top_trials": sorted(trials, key=lambda row: tuple(-float(value) for value in row["selection_key"]))[:10],
        "train_rows": int(train_x.shape[0]),
        "train_positive_rows": int(train_y.sum()),
        "final_train_rows": int(final_x.shape[0]),
        "final_train_positive_rows": int(final_y.sum()),
        "feature_importance": sorted(
            [
                {"feature": name, "importance": float(value)}
                for name, value in zip(feature_names, final_model.feature_importances_, strict=False)
            ],
            key=lambda row: float(row["importance"]),
            reverse=True,
        ),
    }
    return final_model, best_model, feature_names, diagnostics
