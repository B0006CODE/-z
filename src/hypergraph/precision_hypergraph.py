from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from src.knowledge.mesh_hierarchy import (
    MeshDescriptor,
    ancestor_trees,
    descriptor_depth,
    descriptor_tree_numbers,
    hierarchy_feature_values,
    parent_tree,
    tree_depth,
    tree_distance,
)


@dataclass(frozen=True)
class PrecisionHypergraphConfig:
    top_k: int = 100
    seed_passages: int = 10
    max_passage_entities: int = 48
    max_passage_mesh: int = 32
    broad_mesh_depth: int = 2
    specific_mesh_depth: int = 3
    max_anchor_df_ratio: float = 0.35
    excessive_cluster_ratio: float = 0.30
    rrf_k: float = 60.0


def entity_ids(rows: list[dict[str, Any]], max_entities: int | None = None) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        entity_id = str(row.get("entity_id", "")).strip()
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        values.append(entity_id)
        if max_entities is not None and len(values) >= max_entities:
            break
    return values


def mesh_ids(rows: list[dict[str, Any]], max_terms: int | None = None) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        mesh_ui = str(row.get("mesh_ui", "")).strip()
        if not mesh_ui or mesh_ui in seen:
            continue
        seen.add(mesh_ui)
        values.append(mesh_ui)
        if max_terms is not None and len(values) >= max_terms:
            break
    return values


def source_score(row: dict[str, Any], source: str, default: float = 0.0) -> float:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        return default
    for candidate in [
        metadata,
        metadata.get("source_metadata", {}),
        metadata.get("base_source_metadata", {}),
    ]:
        if not isinstance(candidate, dict):
            continue
        value = candidate.get("source_scores", {}).get(source)
        if value is not None:
            return float(value)
    return default


def semantic_score(row: dict[str, Any], semantic_lookup: dict[tuple[str, str], float] | None = None) -> float:
    qid = str(row["question_id"])
    pid = str(row["passage_id"])
    if semantic_lookup and (qid, pid) in semantic_lookup:
        return float(semantic_lookup[(qid, pid)])
    medcpt = source_score(row, "medcpt", None)  # type: ignore[arg-type]
    if medcpt is not None:
        return float(medcpt)
    return source_score(row, "dense", float(row.get("score", 0.0)))


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _mesh_depth(mesh_ui: str, hierarchy: dict[str, MeshDescriptor]) -> int:
    return descriptor_depth(mesh_ui, hierarchy)


def _tree_relatedness(
    q_mesh: set[str],
    p_mesh: set[str],
    hierarchy: dict[str, MeshDescriptor],
    *,
    min_specific_depth: int,
) -> tuple[float, int, set[str]]:
    best_scores: list[float] = []
    matched = 0
    concepts: set[str] = set()
    for q_ui in q_mesh:
        for q_tree in descriptor_tree_numbers(q_ui, hierarchy):
            q_parent = parent_tree(q_tree)
            q_ancestors = ancestor_trees(q_tree, include_self=True)
            for p_ui in p_mesh:
                for p_tree in descriptor_tree_numbers(p_ui, hierarchy):
                    p_depth = tree_depth(p_tree)
                    if p_depth < min_specific_depth:
                        continue
                    distance = tree_distance(q_tree, p_tree)
                    parent_match = q_parent is not None and q_parent == parent_tree(p_tree) and q_tree != p_tree
                    ancestor_match = p_tree in q_ancestors or q_tree in ancestor_trees(p_tree, include_self=True)
                    if distance is None and not parent_match and not ancestor_match:
                        continue
                    matched += 1
                    if distance is None:
                        value = 0.25
                    else:
                        value = 1.0 / (1.0 + float(distance))
                    if ancestor_match:
                        value = max(value, 0.35)
                    if parent_match:
                        value = max(value, 0.30)
                    value *= min(p_depth / 6.0, 1.25)
                    best_scores.append(value)
                    concepts.add(f"mesh_tree:{p_tree}")
    return (sum(best_scores) / len(best_scores) if best_scores else 0.0, matched, concepts)


def _relation_targets(
    q_entities: set[str],
    entity_relations: dict[str, list[dict[str, Any]]],
) -> set[str]:
    targets: set[str] = set()
    for entity_id in q_entities:
        for relation in entity_relations.get(entity_id, []):
            target = str(relation.get("target_entity_id", "")).strip()
            if target:
                targets.add(target)
    return targets


def _idf(count: int, total: int) -> float:
    return math.log1p((total + 1.0) / (count + 1.0))


def _candidate_concepts(
    p_entities: set[str],
    p_mesh: set[str],
    p_hierarchy_concepts: set[str],
) -> set[str]:
    return {f"entity:{eid}" for eid in p_entities} | {f"mesh:{ui}" for ui in p_mesh} | set(p_hierarchy_concepts)


def build_precision_hypergraph_features(
    question_id: str,
    candidates: list[dict[str, Any]],
    question_entities: list[dict[str, Any]],
    passage_entities: dict[str, list[dict[str, Any]]],
    question_mesh: list[dict[str, Any]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    mesh_hierarchy: dict[str, MeshDescriptor],
    entity_relations: dict[str, list[dict[str, Any]]],
    semantic_lookup: dict[tuple[str, str], float] | None = None,
    *,
    config: PrecisionHypergraphConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or PrecisionHypergraphConfig()
    candidates = candidates[: cfg.top_k]
    total = max(len(candidates), 1)

    q_entities = set(entity_ids(question_entities))
    q_mesh = set(mesh_ids(question_mesh))
    q_relation_targets = _relation_targets(q_entities, entity_relations)
    query_anchor_count = len(q_entities) + len(q_mesh) + len(q_relation_targets)

    p_entities_by_pid: dict[str, set[str]] = {}
    p_mesh_by_pid: dict[str, set[str]] = {}
    entity_df: Counter[str] = Counter()
    mesh_df: Counter[str] = Counter()
    hierarchy_concepts_by_pid: dict[str, set[str]] = {}

    for row in candidates:
        pid = str(row["passage_id"])
        p_entities = set(entity_ids(passage_entities.get(pid, []), cfg.max_passage_entities))
        p_mesh = set(mesh_ids(passage_mesh.get(pid, []), cfg.max_passage_mesh))
        p_entities_by_pid[pid] = p_entities
        p_mesh_by_pid[pid] = p_mesh
        entity_df.update(p_entities)
        mesh_df.update(p_mesh)
        hierarchy_concepts: set[str] = set()
        for mesh_ui in p_mesh:
            for tree_number in descriptor_tree_numbers(mesh_ui, mesh_hierarchy):
                if tree_depth(tree_number) >= cfg.specific_mesh_depth:
                    hierarchy_concepts.add(f"mesh_tree:{tree_number}")
                parent = parent_tree(tree_number)
                if parent and tree_depth(parent) >= cfg.specific_mesh_depth:
                    hierarchy_concepts.add(f"mesh_parent:{parent}")
        hierarchy_concepts_by_pid[pid] = hierarchy_concepts

    semantic_values = [semantic_score(row, semantic_lookup) for row in candidates]
    semantic_norm = {
        str(row["passage_id"]): value
        for row, value in zip(candidates, minmax(semantic_values), strict=False)
    }

    seed_rows = candidates[: cfg.seed_passages]
    seed_concepts: dict[str, set[str]] = {}
    for row in seed_rows:
        pid = str(row["passage_id"])
        exact_entities = q_entities & p_entities_by_pid.get(pid, set())
        exact_mesh = q_mesh & p_mesh_by_pid.get(pid, set())
        related_entities = q_relation_targets & p_entities_by_pid.get(pid, set())
        if exact_entities or exact_mesh or related_entities:
            seed_concepts[pid] = _candidate_concepts(
                exact_entities | related_entities,
                exact_mesh,
                hierarchy_concepts_by_pid.get(pid, set()),
            )

    rows: list[dict[str, Any]] = []
    raw_scores: list[float] = []
    for row in candidates:
        pid = str(row["passage_id"])
        p_entities = p_entities_by_pid.get(pid, set())
        p_mesh = p_mesh_by_pid.get(pid, set())
        direct_entities = q_entities & p_entities
        direct_mesh = q_mesh & p_mesh
        relation_entities = q_relation_targets & p_entities
        entity_union = q_entities | p_entities
        mesh_union = q_mesh | p_mesh

        rare_entity_score = sum(
            _idf(entity_df.get(entity_id, 0), total)
            for entity_id in direct_entities
            if entity_df.get(entity_id, total) / total <= cfg.max_anchor_df_ratio
        )
        synonym_cluster_score = float(len(direct_entities))
        mesh_specificity_score = 0.0
        broad_mesh_penalty = 0.0
        for mesh_ui in direct_mesh:
            depth = _mesh_depth(mesh_ui, mesh_hierarchy)
            df_ratio = mesh_df.get(mesh_ui, 0) / total
            if depth >= cfg.specific_mesh_depth:
                mesh_specificity_score += _idf(mesh_df.get(mesh_ui, 0), total) * min(depth / 6.0, 1.25)
            if depth <= cfg.broad_mesh_depth or df_ratio > cfg.max_anchor_df_ratio:
                broad_mesh_penalty += 0.5 + df_ratio

        hierarchy_score, hierarchy_edges, hierarchy_concepts = _tree_relatedness(
            q_mesh,
            p_mesh,
            mesh_hierarchy,
            min_specific_depth=cfg.specific_mesh_depth,
        )
        relation_score = sum(_idf(entity_df.get(entity_id, 0), total) for entity_id in relation_entities)

        candidate_anchor_concepts = _candidate_concepts(
            direct_entities | relation_entities,
            direct_mesh,
            hierarchy_concepts,
        )
        seed_support = 0.0
        seed_hits = 0
        for seed_pid, concepts in seed_concepts.items():
            if seed_pid == pid:
                continue
            overlap = candidate_anchor_concepts & concepts
            if overlap:
                seed_hits += 1
                seed_rank = next((int(item["rank"]) for item in seed_rows if str(item["passage_id"]) == seed_pid), cfg.seed_passages)
                seed_support += len(overlap) / (cfg.rrf_k + seed_rank)
        shared_seed_support_score = seed_support * 10.0

        local_df_ratios = [
            entity_df.get(entity_id, 0) / total
            for entity_id in direct_entities | relation_entities
        ] + [
            mesh_df.get(mesh_ui, 0) / total
            for mesh_ui in direct_mesh
        ]
        high_df_penalty = max(local_df_ratios, default=0.0)
        excessive_shared_penalty = max(0.0, high_df_penalty - cfg.excessive_cluster_ratio)
        hierarchy_only = hierarchy_score > 0.0 and not direct_entities and not direct_mesh and not relation_entities
        hierarchy_only_penalty = 1.0 if hierarchy_only else 0.0

        positive_signal = (
            rare_entity_score
            + mesh_specificity_score
            + hierarchy_score
            + shared_seed_support_score
            + 0.5 * synonym_cluster_score
            + relation_score
        )
        sem_agreement = min(positive_signal / 3.0, 1.0) * semantic_norm.get(pid, 0.0)
        sem_disagreement_penalty = max(0.0, min(positive_signal / 3.0, 1.0) - semantic_norm.get(pid, 0.0))
        raw_score = (
            positive_signal
            + 0.5 * sem_agreement
            - 0.7 * broad_mesh_penalty
            - 0.8 * hierarchy_only_penalty
            - 0.5 * high_df_penalty
            - 0.7 * excessive_shared_penalty
            - 0.4 * sem_disagreement_penalty
        )
        raw_scores.append(raw_score)

        hierarchy_values = hierarchy_feature_values(
            question_mesh,
            passage_mesh.get(pid, []),
            mesh_hierarchy,
            max_passage_mesh=cfg.max_passage_mesh,
        )
        local_hyperedge_count = (
            len(direct_entities)
            + len(direct_mesh)
            + hierarchy_edges
            + len(relation_entities)
            + seed_hits
        )
        features = {
            "precision_hypergraph_raw_score": float(raw_score),
            "rare_entity_bridge_score": float(rare_entity_score),
            "mesh_specificity_score": float(mesh_specificity_score),
            "mesh_hierarchy_bridge_score": float(hierarchy_score),
            "shared_seed_support_score": float(shared_seed_support_score),
            "semantic_graph_agreement": float(sem_agreement),
            "semantic_graph_disagreement_penalty": float(sem_disagreement_penalty),
            "broad_concept_penalty": float(broad_mesh_penalty),
            "hierarchy_only_penalty": float(hierarchy_only_penalty),
            "high_df_concept_penalty": float(high_df_penalty),
            "excessive_shared_cluster_penalty": float(excessive_shared_penalty),
            "local_hyperedge_count": float(local_hyperedge_count),
            "query_anchor_count": float(query_anchor_count),
            "synonym_cluster_score": float(synonym_cluster_score),
            "relation_tuple_score": float(relation_score),
            "direct_entity_overlap_count": float(len(direct_entities)),
            "direct_mesh_overlap_count": float(len(direct_mesh)),
            "relation_entity_overlap_count": float(len(relation_entities)),
            "entity_jaccard": len(direct_entities) / len(entity_union) if entity_union else 0.0,
            "question_entity_coverage": len(direct_entities) / len(q_entities) if q_entities else 0.0,
            "mesh_jaccard": len(direct_mesh) / len(mesh_union) if mesh_union else 0.0,
            "question_mesh_coverage": len(direct_mesh) / len(q_mesh) if q_mesh else 0.0,
            "semantic_score_norm": float(semantic_norm.get(pid, 0.0)),
            "base_rank_score": float(1.0 / (cfg.rrf_k + int(row.get("rank", total)))),
            "hybrid_score": float(row.get("score", 0.0)),
            "bm25_score": source_score(row, "bm25"),
            "dense_score": source_score(row, "dense"),
            "medcpt_score": source_score(row, "medcpt"),
            "biomedical_semantic_score": float(semantic_values[len(rows)] if len(rows) < len(semantic_values) else 0.0),
            "rank_percentile": 1.0 - ((int(row.get("rank", total)) - 1) / max(total - 1, 1)),
            **hierarchy_values,
        }
        rows.append(
            {
                "row": row,
                "base_rank": int(row.get("rank", total)),
                "features": features,
                "anchor_concepts": sorted(candidate_anchor_concepts),
            }
        )

    normalized = minmax(raw_scores)
    for item, score in zip(rows, normalized, strict=False):
        item["features"]["precision_hypergraph_score"] = float(score)
    return rows


def build_precision_features_by_qid(
    predictions_by_qid: dict[str, list[dict[str, Any]]],
    question_entities: dict[str, list[dict[str, Any]]],
    passage_entities: dict[str, list[dict[str, Any]]],
    question_mesh: dict[str, list[dict[str, Any]]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    mesh_hierarchy: dict[str, MeshDescriptor],
    entity_relations: dict[str, list[dict[str, Any]]],
    semantic_lookup: dict[tuple[str, str], float] | None = None,
    *,
    config: PrecisionHypergraphConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for qid in sorted(predictions_by_qid):
        output[qid] = build_precision_hypergraph_features(
            qid,
            predictions_by_qid[qid],
            question_entities.get(qid, []),
            passage_entities,
            question_mesh.get(qid, []),
            passage_mesh,
            mesh_hierarchy,
            entity_relations,
            semantic_lookup,
            config=config,
        )
    return output
