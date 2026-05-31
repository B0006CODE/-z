from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SlateWeights:
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    delta: float = 0.0
    enabled: bool = False


def candidate_slate_weights() -> list[SlateWeights]:
    weights = [SlateWeights(enabled=False)]
    for alpha in [0.0, 0.05, 0.10]:
        for beta in [0.0, 0.05, 0.10]:
            for gamma in [0.0, 0.03, 0.06]:
                for delta in [0.0, 0.03, 0.06]:
                    if alpha == beta == gamma == delta == 0.0:
                        continue
                    weights.append(SlateWeights(alpha=alpha, beta=beta, gamma=gamma, delta=delta, enabled=True))
    return weights


def _feature(item: dict[str, Any], name: str) -> float:
    return float(item.get("features", {}).get(name, 0.0))


def optimize_slate(
    scored_items: list[tuple[float, dict[str, Any]]],
    *,
    weights: SlateWeights,
    slate_k: int = 10,
    pool_size: int = 30,
) -> list[tuple[float, dict[str, Any]]]:
    if not weights.enabled:
        return sorted(scored_items, key=lambda pair: (-pair[0], int(pair[1].get("base_rank", 999999))))[:slate_k]

    pool = sorted(scored_items, key=lambda pair: (-pair[0], int(pair[1].get("base_rank", 999999))))[:pool_size]
    remaining = list(pool)
    selected: list[tuple[float, dict[str, Any]]] = []
    covered_concepts: set[str] = set()

    while remaining and len(selected) < slate_k:
        best_idx = 0
        best_utility = float("-inf")
        for idx, (score, item) in enumerate(remaining):
            features = item.get("features", {})
            concepts = set(item.get("anchor_concepts", []))
            new_concepts = concepts - covered_concepts
            reliable_gain = float(features.get("cf_total_positive_gain", 0.0))
            coverage_gain = len(new_concepts) / max(float(features.get("query_anchor_count", 1.0)), 1.0)
            redundant_penalty = len(concepts & covered_concepts) / max(len(concepts), 1)
            broad_noise = (
                float(features.get("broad_concept_penalty", 0.0))
                + float(features.get("high_df_concept_penalty", 0.0))
                + float(features.get("excessive_shared_cluster_penalty", 0.0))
            )
            utility = (
                score
                + weights.alpha * reliable_gain
                + weights.beta * coverage_gain
                - weights.gamma * redundant_penalty
                - weights.delta * broad_noise
            )
            if utility > best_utility:
                best_utility = utility
                best_idx = idx
        score, item = remaining.pop(best_idx)
        selected.append((float(best_utility), item))
        covered_concepts.update(item.get("anchor_concepts", []))

    if len(selected) < slate_k:
        selected.extend(remaining[: slate_k - len(selected)])
    return selected
