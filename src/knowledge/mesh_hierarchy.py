from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MeshDescriptor:
    mesh_ui: str
    mesh_name: str
    tree_numbers: tuple[str, ...]


def load_mesh_hierarchy(rows: list[dict[str, Any]]) -> dict[str, MeshDescriptor]:
    hierarchy: dict[str, MeshDescriptor] = {}
    for row in rows:
        mesh_ui = str(row.get("mesh_ui", "")).strip()
        if not mesh_ui:
            continue
        tree_numbers = tuple(
            str(value).strip()
            for value in row.get("tree_numbers", [])
            if str(value).strip()
        )
        hierarchy[mesh_ui] = MeshDescriptor(
            mesh_ui=mesh_ui,
            mesh_name=str(row.get("mesh_name", "")).strip(),
            tree_numbers=tree_numbers,
        )
    return hierarchy


def mesh_ids(mesh_rows: list[dict[str, Any]], max_terms: int | None = None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for term in mesh_rows:
        mesh_ui = str(term.get("mesh_ui", "")).strip()
        if not mesh_ui or mesh_ui in seen:
            continue
        seen.add(mesh_ui)
        ids.append(mesh_ui)
        if max_terms is not None and len(ids) >= max_terms:
            break
    return ids


def tree_depth(tree_number: str) -> int:
    return len([part for part in tree_number.split(".") if part])


def parent_tree(tree_number: str) -> str | None:
    parts = [part for part in tree_number.split(".") if part]
    if len(parts) <= 1:
        return None
    return ".".join(parts[:-1])


def ancestor_trees(tree_number: str, *, include_self: bool = False) -> set[str]:
    parts = [part for part in tree_number.split(".") if part]
    stop = len(parts) if include_self else len(parts) - 1
    return {".".join(parts[:idx]) for idx in range(1, stop + 1)}


def lca_depth(left: str, right: str) -> int:
    left_parts = left.split(".")
    right_parts = right.split(".")
    depth = 0
    for l_part, r_part in zip(left_parts, right_parts, strict=False):
        if l_part != r_part:
            break
        depth += 1
    return depth


def tree_distance(left: str, right: str) -> int | None:
    shared = lca_depth(left, right)
    if shared == 0:
        return None
    return tree_depth(left) + tree_depth(right) - 2 * shared


def descriptor_tree_numbers(mesh_ui: str, hierarchy: dict[str, MeshDescriptor]) -> tuple[str, ...]:
    descriptor = hierarchy.get(mesh_ui)
    return descriptor.tree_numbers if descriptor else ()


def descriptor_depth(mesh_ui: str, hierarchy: dict[str, MeshDescriptor]) -> int:
    tree_numbers = descriptor_tree_numbers(mesh_ui, hierarchy)
    if not tree_numbers:
        return 0
    return max(tree_depth(tree_number) for tree_number in tree_numbers)


def build_parent_child_index(hierarchy: dict[str, MeshDescriptor]) -> dict[str, set[str]]:
    parent_to_children: dict[str, set[str]] = defaultdict(set)
    for descriptor in hierarchy.values():
        for tree_number in descriptor.tree_numbers:
            parent = parent_tree(tree_number)
            if parent:
                parent_to_children[parent].add(tree_number)
    return dict(parent_to_children)


def hierarchy_feature_values(
    question_mesh_rows: list[dict[str, Any]],
    passage_mesh_rows: list[dict[str, Any]],
    hierarchy: dict[str, MeshDescriptor],
    *,
    max_question_mesh: int | None = None,
    max_passage_mesh: int | None = None,
) -> dict[str, float]:
    q_ids = mesh_ids(question_mesh_rows, max_question_mesh)
    p_ids = mesh_ids(passage_mesh_rows, max_passage_mesh)
    q_set = set(q_ids)
    p_set = set(p_ids)
    q_tree_by_ui = {mesh_ui: descriptor_tree_numbers(mesh_ui, hierarchy) for mesh_ui in q_ids}
    p_tree_by_ui = {mesh_ui: descriptor_tree_numbers(mesh_ui, hierarchy) for mesh_ui in p_ids}
    q_trees = [tree for trees in q_tree_by_ui.values() for tree in trees]
    p_trees = [tree for trees in p_tree_by_ui.values() for tree in trees]

    exact_overlap = q_set & p_set
    ancestor_matches: set[str] = set()
    parent_matches: set[str] = set()
    sibling_matches: set[str] = set()
    covered_p: set[str] = set()
    best_similarities: list[float] = []
    best_distances: list[int] = []

    p_tree_set = set(p_trees)
    p_parent_set = {parent for tree in p_trees if (parent := parent_tree(tree))}
    for q_ui, q_tree_numbers in q_tree_by_ui.items():
        best_similarity = 0.0
        best_distance: int | None = None
        matched_ancestor = False
        matched_parent = False
        matched_sibling = False
        for q_tree in q_tree_numbers:
            q_parent = parent_tree(q_tree)
            q_ancestors = ancestor_trees(q_tree, include_self=False)
            if q_parent and q_parent in p_tree_set:
                matched_parent = True
            if q_ancestors & p_tree_set or q_tree in p_parent_set:
                matched_ancestor = True
            for p_ui, p_tree_numbers in p_tree_by_ui.items():
                for p_tree in p_tree_numbers:
                    distance = tree_distance(q_tree, p_tree)
                    if distance is None:
                        continue
                    if distance <= 2:
                        covered_p.add(p_ui)
                    similarity = 1.0 / (1.0 + distance)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_distance = distance
                    if q_parent and q_parent == parent_tree(p_tree) and q_tree != p_tree:
                        matched_sibling = True
        if matched_parent:
            parent_matches.add(q_ui)
        if matched_ancestor:
            ancestor_matches.add(q_ui)
        if matched_sibling:
            sibling_matches.add(q_ui)
        if best_similarity > 0:
            best_similarities.append(best_similarity)
        if best_distance is not None:
            best_distances.append(best_distance)

    p_depths = [descriptor_depth(mesh_ui, hierarchy) for mesh_ui in p_ids]
    p_depths = [depth for depth in p_depths if depth > 0]
    covered_q = exact_overlap | ancestor_matches | parent_matches | sibling_matches
    covered_p |= exact_overlap

    return {
        "mesh_hierarchy_exact_count": float(len(exact_overlap)),
        "mesh_parent_match_count": float(len(parent_matches)),
        "mesh_ancestor_match_count": float(len(ancestor_matches)),
        "mesh_sibling_match_count": float(len(sibling_matches)),
        "mesh_tree_similarity_max": max(best_similarities) if best_similarities else 0.0,
        "mesh_tree_similarity_mean": sum(best_similarities) / len(best_similarities) if best_similarities else 0.0,
        "mesh_tree_distance_min": float(min(best_distances)) if best_distances else 0.0,
        "question_mesh_hierarchy_coverage": len(covered_q) / len(q_set) if q_set else 0.0,
        "passage_mesh_hierarchy_coverage": len(covered_p) / len(p_set) if p_set else 0.0,
        "passage_mesh_specificity": sum(p_depths) / len(p_depths) if p_depths else 0.0,
    }


def shared_mesh_cluster_features(
    candidate_mesh_rows: dict[str, list[dict[str, Any]]],
    hierarchy: dict[str, MeshDescriptor],
    *,
    max_passage_mesh: int | None = None,
) -> dict[str, dict[str, float]]:
    term_counts: dict[str, int] = defaultdict(int)
    parent_counts: dict[str, int] = defaultdict(int)
    candidate_terms: dict[str, set[str]] = {}
    candidate_parents: dict[str, set[str]] = {}

    for passage_id, rows in candidate_mesh_rows.items():
        terms = set(mesh_ids(rows, max_passage_mesh))
        parents = {
            parent
            for mesh_ui in terms
            for tree_number in descriptor_tree_numbers(mesh_ui, hierarchy)
            if (parent := parent_tree(tree_number))
        }
        candidate_terms[passage_id] = terms
        candidate_parents[passage_id] = parents
        for term in terms:
            term_counts[term] += 1
        for parent in parents:
            parent_counts[parent] += 1

    features: dict[str, dict[str, float]] = {}
    total_candidates = max(len(candidate_mesh_rows), 1)
    for passage_id in candidate_mesh_rows:
        term_cluster = max((term_counts[term] for term in candidate_terms.get(passage_id, set())), default=0)
        parent_cluster = max((parent_counts[parent] for parent in candidate_parents.get(passage_id, set())), default=0)
        features[passage_id] = {
            "shared_mesh_term_cluster_size": float(term_cluster),
            "shared_mesh_parent_cluster_size": float(parent_cluster),
            "shared_mesh_term_cluster_ratio": term_cluster / total_candidates,
            "shared_mesh_parent_cluster_ratio": parent_cluster / total_candidates,
        }
    return features
