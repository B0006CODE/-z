from __future__ import annotations

from typing import Any


COUNTERFACTUAL_FEATURES = [
    "full_hypergraph_score",
    "score_without_rare_entity_edges",
    "score_without_mesh_hierarchy_edges",
    "score_without_shared_seed_edges",
    "score_without_synonym_cluster_edges",
    "score_without_relation_edges",
    "score_without_semantic_agreement",
    "cf_gain_rare_entity",
    "cf_gain_mesh_hierarchy",
    "cf_gain_shared_seed",
    "cf_gain_synonym_cluster",
    "cf_gain_relation",
    "cf_gain_semantic_agreement",
    "cf_total_positive_gain",
    "cf_max_edge_type_gain",
]


def _positive(value: Any) -> float:
    return max(float(value or 0.0), 0.0)


def add_counterfactual_features(features_by_qid: dict[str, list[dict[str, Any]]]) -> None:
    for items in features_by_qid.values():
        for item in items:
            features = item["features"]
            full = float(features.get("precision_hypergraph_score", 0.0))
            gains = {
                "rare_entity": _positive(features.get("rare_entity_bridge_score")),
                "mesh_hierarchy": _positive(features.get("mesh_hierarchy_bridge_score")),
                "shared_seed": _positive(features.get("shared_seed_support_score")),
                "synonym_cluster": _positive(features.get("synonym_cluster_score")),
                "relation": _positive(features.get("relation_tuple_score")),
                "semantic_agreement": _positive(features.get("semantic_graph_agreement")),
            }
            total_raw = sum(gains.values())
            if total_raw > 0:
                scaled = {name: full * value / total_raw for name, value in gains.items()}
            else:
                scaled = {name: 0.0 for name in gains}

            features["full_hypergraph_score"] = full
            features["score_without_rare_entity_edges"] = full - scaled["rare_entity"]
            features["score_without_mesh_hierarchy_edges"] = full - scaled["mesh_hierarchy"]
            features["score_without_shared_seed_edges"] = full - scaled["shared_seed"]
            features["score_without_synonym_cluster_edges"] = full - scaled["synonym_cluster"]
            features["score_without_relation_edges"] = full - scaled["relation"]
            features["score_without_semantic_agreement"] = full - scaled["semantic_agreement"]
            features["cf_gain_rare_entity"] = scaled["rare_entity"]
            features["cf_gain_mesh_hierarchy"] = scaled["mesh_hierarchy"]
            features["cf_gain_shared_seed"] = scaled["shared_seed"]
            features["cf_gain_synonym_cluster"] = scaled["synonym_cluster"]
            features["cf_gain_relation"] = scaled["relation"]
            features["cf_gain_semantic_agreement"] = scaled["semantic_agreement"]
            features["cf_total_positive_gain"] = sum(scaled.values())
            features["cf_max_edge_type_gain"] = max(scaled.values(), default=0.0)
