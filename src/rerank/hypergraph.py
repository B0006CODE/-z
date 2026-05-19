from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.hypergraph.local import hypergraph_features


def group_predictions(predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["question_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["rank"]))
    return dict(grouped)


def entity_map(rows: list[dict[str, Any]], id_key: str) -> dict[str, list[dict[str, Any]]]:
    return {str(row[id_key]): list(row.get("entities", [])) for row in rows}


def mesh_map(rows: list[dict[str, Any]], id_key: str) -> dict[str, list[dict[str, Any]]]:
    return {str(row[id_key]): list(row.get("mesh_terms", [])) for row in rows}


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def build_feature_rows(
    predictions: list[dict[str, Any]],
    question_entities: dict[str, list[dict[str, Any]]],
    passage_entities: dict[str, list[dict[str, Any]]],
    question_mesh: dict[str, list[dict[str, Any]]] | None = None,
    passage_mesh: dict[str, list[dict[str, Any]]] | None = None,
    *,
    structure: str = "knowledge_hypergraph",
    top_k: int = 100,
    rrf_k: int = 60,
    iterations: int = 3,
    damping: float = 0.85,
    max_passage_entities: int = 48,
    max_passage_mesh: int = 32,
) -> dict[str, list[dict[str, Any]]]:
    preds_by_qid = group_predictions(predictions)
    features_by_qid: dict[str, list[dict[str, Any]]] = {}
    question_mesh = question_mesh or {}
    passage_mesh = passage_mesh or {}

    for qid in sorted(preds_by_qid):
        candidates = preds_by_qid[qid][:top_k]
        hg_features = hypergraph_features(
            qid,
            candidates,
            question_entities.get(qid, []),
            passage_entities,
            question_mesh=question_mesh.get(qid, []),
            passage_mesh=passage_mesh,
            structure=structure,
            iterations=iterations,
            damping=damping,
            max_passage_entities=max_passage_entities,
            max_passage_mesh=max_passage_mesh,
        )
        base_rank_scores = [1.0 / (rrf_k + int(row["rank"])) for row in candidates]
        base_norm = minmax(base_rank_scores)
        hg_norm = minmax([hg_features.get(str(row["passage_id"]), {}).get("hypergraph_score", 0.0) for row in candidates])

        rows = []
        for row, base_score, normalized_hg in zip(candidates, base_norm, hg_norm, strict=False):
            passage_id = str(row["passage_id"])
            features = dict(hg_features.get(passage_id, {}))
            features["base_rank_score"] = float(base_score)
            features["hypergraph_score_norm"] = float(normalized_hg)
            rows.append(
                {
                    "row": row,
                    "features": features,
                    "base_rank": int(row["rank"]),
                }
            )
        features_by_qid[qid] = rows
    return features_by_qid


def rerank_from_features(
    features_by_qid: dict[str, list[dict[str, Any]]],
    *,
    top_k: int = 100,
    base_weight: float = 1.0,
    hypergraph_weight: float = 0.0,
    entity_weight: float = 0.0,
    mesh_weight: float = 0.0,
    retriever_name: str = "local_hypergraph_rerank",
) -> list[dict[str, Any]]:
    reranked: list[dict[str, Any]] = []
    for qid in sorted(features_by_qid):
        scored = []
        for item in features_by_qid[qid]:
            features = item["features"]
            score = (
                base_weight * float(features.get("base_rank_score", 0.0))
                + hypergraph_weight * float(features.get("hypergraph_score_norm", 0.0))
                + entity_weight * float(features.get("question_entity_coverage", 0.0))
                + mesh_weight * float(features.get("question_mesh_coverage", 0.0))
            )
            scored.append((score, item))

        scored.sort(
            key=lambda pair: (
                -pair[0],
                int(pair[1]["base_rank"]),
                str(pair[1]["row"]["passage_id"]),
            )
        )
        for rank, (score, item) in enumerate(scored[:top_k], start=1):
            row = item["row"]
            reranked.append(
                {
                    "question_id": row["question_id"],
                    "passage_id": row["passage_id"],
                    "rank": rank,
                    "score": float(score),
                    "retriever": retriever_name,
                    "metadata": {
                        "base_rank": item["base_rank"],
                        "base_weight": base_weight,
                        "hypergraph_weight": hypergraph_weight,
                        "entity_weight": entity_weight,
                        "mesh_weight": mesh_weight,
                        "features": item["features"],
                        "source_metadata": row.get("metadata", {}),
                    },
                }
            )
    return reranked
