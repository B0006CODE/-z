from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from src.knowledge.mesh_hierarchy import ancestor_trees, descriptor_tree_numbers, parent_tree, tree_distance


@dataclass(frozen=True)
class HyperPathConfig:
    max_passage_entities: int = 64
    max_passage_mesh: int = 48
    top_seed_n: int = 10
    max_df_ratio: float = 0.45
    exact_mesh_weight: float = 2.0
    hierarchy_weight: float = 1.1
    entity_weight: float = 1.5
    relation_weight: float = 0.8
    seed_shared_weight: float = 1.8
    diffusion_direct_weight: float = 0.65
    diffusion_shared_weight: float = 0.35
    rrf_k: float = 60.0


def entity_ids(rows: list[dict[str, Any]], max_terms: int | None = None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        entity_id = str(row.get("entity_id", "")).strip()
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        ids.append(entity_id)
        if max_terms is not None and len(ids) >= max_terms:
            break
    return ids


def mesh_ids(rows: list[dict[str, Any]], max_terms: int | None = None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        mesh_ui = str(row.get("mesh_ui", "")).strip()
        if not mesh_ui or mesh_ui in seen:
            continue
        seen.add(mesh_ui)
        ids.append(mesh_ui)
        if max_terms is not None and len(ids) >= max_terms:
            break
    return ids


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 if value > 0 else 0.0 for value in values]
    return [(value - low) / (high - low) for value in values]


def _idf(term: str, df: Counter[str], n: int, max_df_ratio: float) -> float:
    count = df.get(term, 0)
    if n > 0 and count / n > max_df_ratio:
        return 0.0
    return math.log((n + 1.0) / (count + 1.0)) + 1.0


def _mesh_tree_set(mesh_set: set[str], hierarchy: dict[str, Any]) -> set[str]:
    trees: set[str] = set()
    for mesh_ui in mesh_set:
        trees.update(descriptor_tree_numbers(mesh_ui, hierarchy))
    return trees


def _mesh_parent_set(mesh_set: set[str], hierarchy: dict[str, Any]) -> set[str]:
    parents: set[str] = set()
    for tree in _mesh_tree_set(mesh_set, hierarchy):
        parent = parent_tree(tree)
        if parent:
            parents.add(parent)
    return parents


def _concepts_for_passage(entity_set: set[str], mesh_set: set[str], hierarchy: dict[str, Any]) -> set[str]:
    concepts = {f"e:{entity_id}" for entity_id in entity_set}
    concepts.update(f"m:{mesh_ui}" for mesh_ui in mesh_set)
    for mesh_ui in mesh_set:
        for tree in descriptor_tree_numbers(mesh_ui, hierarchy):
            parent = parent_tree(tree)
            if parent:
                concepts.add(f"mp:{parent}")
            concepts.update(f"ma:{ancestor}" for ancestor in ancestor_trees(tree, include_self=False))
    return concepts


def _hierarchy_score(
    q_mesh_set: set[str],
    p_mesh_set: set[str],
    hierarchy: dict[str, Any],
    concept_df: Counter[str],
    n: int,
    max_df_ratio: float,
) -> float:
    score = 0.0
    q_trees = [(ui, tree) for ui in q_mesh_set for tree in descriptor_tree_numbers(ui, hierarchy)]
    p_trees = [(ui, tree) for ui in p_mesh_set for tree in descriptor_tree_numbers(ui, hierarchy)]
    matched: set[tuple[str, str, str]] = set()
    for q_ui, q_tree in q_trees:
        q_parent = parent_tree(q_tree)
        q_ancestors = ancestor_trees(q_tree, include_self=False)
        for p_ui, p_tree in p_trees:
            if q_ui == p_ui:
                continue
            p_parent = parent_tree(p_tree)
            dist = tree_distance(q_tree, p_tree)
            if dist is not None and dist <= 3:
                key = ("dist", q_ui, p_ui)
                if key not in matched:
                    matched.add(key)
                    score += (1.0 / (1.0 + dist)) * _idf(f"m:{p_ui}", concept_df, n, max_df_ratio)
            if q_parent and q_parent == p_parent and q_tree != p_tree:
                key = ("sib", q_ui, p_ui)
                if key not in matched:
                    matched.add(key)
                    score += 0.35 * _idf(f"mp:{q_parent}", concept_df, n, max_df_ratio)
            if p_tree in q_ancestors or q_tree in ancestor_trees(p_tree, include_self=False):
                key = ("anc", q_ui, p_ui)
                if key not in matched:
                    matched.add(key)
                    score += 0.55 * _idf(f"m:{p_ui}", concept_df, n, max_df_ratio)
    return score


def compute_hyperpath_features(
    question_id: str,
    candidates: list[dict[str, Any]],
    question_entities: list[dict[str, Any]],
    passage_entities: dict[str, list[dict[str, Any]]],
    question_mesh: list[dict[str, Any]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    mesh_hierarchy: dict[str, Any],
    entity_relations: dict[str, list[dict[str, Any]]],
    *,
    config: HyperPathConfig = HyperPathConfig(),
    include_seed_shared: bool = True,
    include_diffusion: bool = True,
) -> dict[str, dict[str, Any]]:
    candidates = sorted(candidates, key=lambda row: int(row.get("rank", 10**9)))
    candidate_ids = [str(row["passage_id"]) for row in candidates]
    n = len(candidate_ids)
    q_entities = set(entity_ids(question_entities))
    q_mesh = set(mesh_ids(question_mesh))
    relation_targets: set[str] = set()
    for source in q_entities:
        for relation in entity_relations.get(source, []):
            target = str(relation.get("target_entity_id", "")).strip()
            if target:
                relation_targets.add(target)

    p_entities: dict[str, set[str]] = {}
    p_mesh: dict[str, set[str]] = {}
    p_concepts: dict[str, set[str]] = {}
    concept_df: Counter[str] = Counter()
    for pid in candidate_ids:
        p_entities[pid] = set(entity_ids(passage_entities.get(pid, []), config.max_passage_entities))
        p_mesh[pid] = set(mesh_ids(passage_mesh.get(pid, []), config.max_passage_mesh))
        p_concepts[pid] = _concepts_for_passage(p_entities[pid], p_mesh[pid], mesh_hierarchy)
        concept_df.update(p_concepts[pid])

    seed_ids = candidate_ids[: config.top_seed_n]
    seed_concepts_by_pid = {pid: p_concepts.get(pid, set()) for pid in seed_ids}
    seed_prior_by_pid = {
        pid: 1.0 / (config.rrf_k + int(candidates[idx].get("rank", idx + 1)))
        for idx, pid in enumerate(seed_ids)
    }

    raw_rows: dict[str, dict[str, Any]] = {}
    for row in candidates:
        pid = str(row["passage_id"])
        entity_overlap = q_entities & p_entities[pid]
        mesh_overlap = q_mesh & p_mesh[pid]
        related_entities = relation_targets & p_entities[pid]
        exact_mesh = sum(_idf(f"m:{ui}", concept_df, n, config.max_df_ratio) for ui in mesh_overlap)
        hierarchy = _hierarchy_score(q_mesh, p_mesh[pid], mesh_hierarchy, concept_df, n, config.max_df_ratio)
        entity_cluster = sum(
            _idf(f"e:{entity_id}", concept_df, n, config.max_df_ratio)
            * (1.0 + math.log1p(max(concept_df.get(f"e:{entity_id}", 1) - 1, 0)))
            for entity_id in entity_overlap
        )
        relation = sum(_idf(f"e:{entity_id}", concept_df, n, config.max_df_ratio) for entity_id in related_entities)
        seed_shared = 0.0
        if include_seed_shared:
            for seed_pid, seed_concepts in seed_concepts_by_pid.items():
                if seed_pid == pid:
                    continue
                shared = p_concepts[pid] & seed_concepts
                seed_weight = seed_prior_by_pid.get(seed_pid, 0.0)
                for concept in shared:
                    seed_shared += seed_weight * _idf(concept, concept_df, n, config.max_df_ratio)

        path_score_by_type = {
            "exact_mesh": config.exact_mesh_weight * exact_mesh,
            "mesh_ancestor_sibling": config.hierarchy_weight * hierarchy,
            "entity_shared_cluster": config.entity_weight * entity_cluster,
            "primekg_relation": config.relation_weight * relation,
            "seed_shared_medical_concept": config.seed_shared_weight * seed_shared,
        }
        hyperpath_score = sum(path_score_by_type.values())
        raw_rows[pid] = {
            "hyperpath_score_raw": hyperpath_score,
            "path_score_by_type": path_score_by_type,
            "coverage": {
                "question_entity_count": len(q_entities),
                "question_mesh_count": len(q_mesh),
                "passage_entity_count": len(p_entities[pid]),
                "passage_mesh_count": len(p_mesh[pid]),
                "entity_overlap_count": len(entity_overlap),
                "mesh_overlap_count": len(mesh_overlap),
                "primekg_related_entity_count": len(related_entities),
                "concept_count": len(p_concepts[pid]),
            },
        }

    hp_norm_values = minmax([raw_rows[pid]["hyperpath_score_raw"] for pid in candidate_ids])
    for pid, value in zip(candidate_ids, hp_norm_values, strict=False):
        raw_rows[pid]["hyperpath_score"] = float(value)

    if include_diffusion:
        concept_to_passages: dict[str, list[str]] = defaultdict(list)
        for pid, concepts in p_concepts.items():
            for concept in concepts:
                if _idf(concept, concept_df, n, config.max_df_ratio) > 0:
                    concept_to_passages[concept].append(pid)

        passage_mass = {
            pid: raw_rows[pid]["hyperpath_score"] + (seed_prior_by_pid.get(pid, 0.0) * 12.0)
            for pid in candidate_ids
        }
        concept_mass: Counter[str] = Counter()
        for pid, concepts in p_concepts.items():
            if not concepts:
                continue
            denom = sum(_idf(concept, concept_df, n, config.max_df_ratio) for concept in concepts)
            if denom <= 0:
                continue
            for concept in concepts:
                weight = _idf(concept, concept_df, n, config.max_df_ratio)
                if weight > 0:
                    concept_mass[concept] += passage_mass[pid] * weight / denom

        diffusion_raw: dict[str, float] = {}
        for pid, concepts in p_concepts.items():
            shared_signal = 0.0
            for concept in concepts:
                neighbors = concept_to_passages.get(concept, [])
                if len(neighbors) <= 1:
                    continue
                shared_signal += concept_mass[concept] * _idf(concept, concept_df, n, config.max_df_ratio) / len(neighbors)
            diffusion_raw[pid] = (
                config.diffusion_direct_weight * raw_rows[pid]["hyperpath_score"]
                + config.diffusion_shared_weight * shared_signal
            )
        diffusion_norm = minmax([diffusion_raw[pid] for pid in candidate_ids])
    else:
        diffusion_norm = [0.0 for _ in candidate_ids]

    for pid, value in zip(candidate_ids, diffusion_norm, strict=False):
        raw_rows[pid]["diffusion_score"] = float(value)
        raw_rows[pid]["local_hyperedge_coverage"] = {
            "num_candidate_passages": n,
            "num_concepts": len(concept_df),
            "num_shared_concepts": sum(1 for count in concept_df.values() if count >= 2),
            "num_seed_passages": len(seed_ids),
        }
    return raw_rows
