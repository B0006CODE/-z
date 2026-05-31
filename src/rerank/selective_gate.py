from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateRule:
    min_graph_signal: float = 0.25
    strong_candidate_margin: float = 0.15
    max_noise: float = 0.65
    easy_top3_support: float = 0.25
    easy_strength: float = 0.15
    medium_strength: float = 0.50
    high_strength: float = 0.85


def candidate_gate_rules() -> list[GateRule]:
    rules: list[GateRule] = []
    for min_signal in [0.15, 0.25, 0.35, 0.50]:
        for max_noise in [0.45, 0.65, 0.85]:
            for easy_strength in [0.0, 0.10, 0.20]:
                for high_strength in [0.65, 0.85, 1.0]:
                    rules.append(
                        GateRule(
                            min_graph_signal=min_signal,
                            max_noise=max_noise,
                            easy_strength=easy_strength,
                            high_strength=high_strength,
                        )
                    )
    return rules


def _feature(item: dict[str, Any], name: str) -> float:
    return float(item.get("features", {}).get(name, 0.0))


def query_gate_stats(items: list[dict[str, Any]]) -> dict[str, float]:
    top3 = items[:3]
    top10 = items[:10]
    top100 = items[:100]
    top10_support = max((_feature(item, "precision_hypergraph_score") for item in top10), default=0.0)
    top3_direct = max(
        (
            _feature(item, "rare_entity_bridge_score")
            + _feature(item, "mesh_specificity_score")
            + _feature(item, "relation_tuple_score")
            for item in top3
        ),
        default=0.0,
    )
    top100_signal = max((_feature(item, "precision_hypergraph_score") for item in top100), default=0.0)
    top100_signal_rank = 999.0
    for idx, item in enumerate(top100, start=1):
        if _feature(item, "precision_hypergraph_score") == top100_signal:
            top100_signal_rank = float(idx)
            break
    noise = max(
        (
            _feature(item, "broad_concept_penalty")
            + _feature(item, "high_df_concept_penalty")
            + _feature(item, "hierarchy_only_penalty")
            + _feature(item, "excessive_shared_cluster_penalty")
            for item in top100
        ),
        default=0.0,
    )
    sem_graph_agreement = max((_feature(item, "semantic_graph_agreement") for item in top10), default=0.0)
    sparse = max((_feature(item, "local_hyperedge_count") for item in top100), default=0.0) <= 0.0
    return {
        "top3_direct_support": top3_direct,
        "top10_graph_support": top10_support,
        "top100_graph_signal": top100_signal,
        "top100_graph_signal_rank": top100_signal_rank,
        "max_noise": noise,
        "semantic_graph_agreement": sem_graph_agreement,
        "is_sparse": 1.0 if sparse else 0.0,
    }


def intervention_for_query(items: list[dict[str, Any]], rule: GateRule) -> tuple[float, str, dict[str, float]]:
    stats = query_gate_stats(items)
    if stats["is_sparse"] > 0.0:
        return 0.0, "fallback_sparse_hypergraph", stats
    if stats["top100_graph_signal"] < rule.min_graph_signal:
        return 0.0, "fallback_weak_query_anchored_signal", stats
    if stats["max_noise"] > rule.max_noise and stats["top3_direct_support"] <= 0.0:
        return 0.0, "fallback_noisy_broad_concepts", stats
    if (
        stats["top3_direct_support"] >= rule.easy_top3_support
        and stats["semantic_graph_agreement"] >= rule.easy_top3_support
        and stats["top100_graph_signal_rank"] <= 3
    ):
        return rule.easy_strength, "easy_top3_supported_low_intervention", stats
    if (
        stats["top10_graph_support"] < rule.min_graph_signal
        and stats["top100_graph_signal"] >= rule.min_graph_signal + rule.strong_candidate_margin
        and stats["top100_graph_signal_rank"] > 10
    ):
        return rule.high_strength, "strong_query_anchored_candidate_beyond_top10", stats
    return rule.medium_strength, "moderate_query_anchored_intervention", stats
