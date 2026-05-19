from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Hyperedge:
    edge_id: str
    edge_type: str
    nodes: tuple[str, ...]
    weight: float = 1.0


@dataclass
class LocalHypergraph:
    nodes: list[str]
    node_types: dict[str, str]
    hyperedges: list[Hyperedge]
    passage_nodes: dict[str, str]
    question_node: str


def _entity_node(entity_id: str) -> str:
    return f"entity:{entity_id}"


def _passage_node(passage_id: str) -> str:
    return f"passage:{passage_id}"


def _type_node(entity_type: str) -> str:
    return f"concept:{entity_type}"


def _entity_ids(entity_rows: list[dict[str, Any]], max_entities: int | None = None) -> list[str]:
    ids = []
    seen = set()
    for entity in entity_rows:
        entity_id = str(entity.get("entity_id", ""))
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        ids.append(entity_id)
        if max_entities is not None and len(ids) >= max_entities:
            break
    return ids


def build_local_hypergraph(
    question_id: str,
    candidates: list[dict[str, Any]],
    question_entities: list[dict[str, Any]],
    passage_entities: dict[str, list[dict[str, Any]]],
    *,
    structure: str = "knowledge_hypergraph",
    max_passage_entities: int = 48,
    max_shared_entities: int = 128,
    candidate_edge_weight: float = 0.15,
    type_edge_weight: float = 0.2,
) -> LocalHypergraph:
    if structure not in {"knowledge_hypergraph", "no_knowledge_hypergraph", "pairwise_graph"}:
        raise ValueError(f"Unsupported local graph structure: {structure}")

    nodes: list[str] = []
    node_types: dict[str, str] = {}
    hyperedges: list[Hyperedge] = []
    passage_nodes: dict[str, str] = {}

    def add_node(node_id: str, node_type: str) -> None:
        if node_id not in node_types:
            node_types[node_id] = node_type
            nodes.append(node_id)

    def add_edge(edge_id: str, edge_type: str, raw_nodes: list[str], weight: float = 1.0) -> None:
        deduped = tuple(dict.fromkeys(node for node in raw_nodes if node in node_types))
        if len(deduped) >= 2:
            if structure == "pairwise_graph" and len(deduped) > 2:
                anchor = deduped[0]
                for idx, node in enumerate(deduped[1:], start=1):
                    hyperedges.append(
                        Hyperedge(edge_id=f"{edge_id}:pair:{idx}", edge_type=edge_type, nodes=(anchor, node), weight=weight)
                    )
            else:
                hyperedges.append(Hyperedge(edge_id=edge_id, edge_type=edge_type, nodes=deduped, weight=weight))

    question_node = f"question:{question_id}"
    add_node(question_node, "question")

    q_entity_ids = _entity_ids(question_entities)
    if structure != "no_knowledge_hypergraph":
        for entity_id in q_entity_ids:
            add_node(_entity_node(entity_id), "entity")
        add_edge(f"qe:{question_id}", "question_entity", [question_node, *(_entity_node(eid) for eid in q_entity_ids)], 1.25)

    entity_to_passages: dict[str, list[str]] = defaultdict(list)
    entity_types: dict[str, str] = {}
    type_to_entities: dict[str, list[str]] = defaultdict(list)

    for row in candidates:
        passage_id = str(row["passage_id"])
        p_node = _passage_node(passage_id)
        passage_nodes[passage_id] = p_node
        add_node(p_node, "passage")
        add_edge(f"qp:{question_id}:{passage_id}", "question_passage_candidate", [question_node, p_node], candidate_edge_weight)

        if structure == "no_knowledge_hypergraph":
            continue

        p_entities = passage_entities.get(passage_id, [])[:max_passage_entities]
        p_entity_ids = _entity_ids(p_entities, max_entities=max_passage_entities)
        for entity in p_entities:
            entity_id = str(entity.get("entity_id", ""))
            if not entity_id:
                continue
            entity_type = str(entity.get("entity_type", "biomedical_concept"))
            entity_types.setdefault(entity_id, entity_type)
        for entity_id in p_entity_ids:
            e_node = _entity_node(entity_id)
            add_node(e_node, "entity")
            entity_to_passages[entity_id].append(passage_id)
            type_to_entities[entity_types.get(entity_id, "biomedical_concept")].append(entity_id)
        add_edge(
            f"pe:{passage_id}",
            "passage_entity",
            [p_node, *(_entity_node(eid) for eid in p_entity_ids)],
            1.0,
        )

    if structure == "no_knowledge_hypergraph":
        rank_bands = [(5, 0.5), (10, 0.35), (20, 0.2), (50, 0.1), (100, 0.05)]
        for band_size, weight in rank_bands:
            band_nodes = [
                _passage_node(str(row["passage_id"]))
                for row in candidates
                if int(row.get("rank", band_size + 1)) <= band_size
            ]
            add_edge(f"rank_band:{question_id}:top{band_size}", "rank_band", [question_node, *band_nodes], weight)
        return LocalHypergraph(
            nodes=nodes,
            node_types=node_types,
            hyperedges=hyperedges,
            passage_nodes=passage_nodes,
            question_node=question_node,
        )

    for entity_type, entity_ids in type_to_entities.items():
        unique_ids = list(dict.fromkeys(entity_ids))
        if len(unique_ids) < 2:
            continue
        t_node = _type_node(entity_type)
        add_node(t_node, "concept")
        add_edge(
            f"type:{entity_type}",
            "entity_type_concept",
            [t_node, *(_entity_node(eid) for eid in unique_ids[:max_shared_entities])],
            type_edge_weight,
        )

    shared = [
        (entity_id, passage_ids)
        for entity_id, passage_ids in entity_to_passages.items()
        if len(set(passage_ids)) >= 2
    ]
    shared.sort(key=lambda item: (-len(set(item[1])), item[0]))
    for entity_id, passage_ids in shared[:max_shared_entities]:
        unique_passage_ids = list(dict.fromkeys(passage_ids))
        add_edge(
            f"shared:{entity_id}",
            "shared_entity_passages",
            [_entity_node(entity_id), *(_passage_node(pid) for pid in unique_passage_ids)],
            0.75,
        )

    return LocalHypergraph(
        nodes=nodes,
        node_types=node_types,
        hyperedges=hyperedges,
        passage_nodes=passage_nodes,
        question_node=question_node,
    )


def diffuse(
    graph: LocalHypergraph,
    seed_nodes: list[str],
    *,
    iterations: int = 3,
    damping: float = 0.85,
) -> dict[str, float]:
    node_index = {node_id: idx for idx, node_id in enumerate(graph.nodes)}
    seed = np.zeros(len(graph.nodes), dtype=np.float64)
    for node_id in seed_nodes:
        idx = node_index.get(node_id)
        if idx is not None:
            seed[idx] += 1.0
    if seed.sum() == 0:
        idx = node_index.get(graph.question_node)
        if idx is not None:
            seed[idx] = 1.0
    seed = seed / seed.sum() if seed.sum() else seed
    scores = seed.copy()

    indexed_edges = [
        ([node_index[node] for node in edge.nodes if node in node_index], max(float(edge.weight), 0.0))
        for edge in graph.hyperedges
    ]
    indexed_edges = [(indices, weight) for indices, weight in indexed_edges if len(indices) >= 2 and weight > 0]

    for _ in range(iterations):
        next_scores = (1.0 - damping) * seed
        for indices, weight in indexed_edges:
            edge_mass = float(scores[indices].mean()) * weight
            if edge_mass <= 0:
                continue
            share = damping * edge_mass / len(indices)
            for idx in indices:
                next_scores[idx] += share
        total = float(next_scores.sum())
        scores = next_scores / total if total > 0 else next_scores

    return {node_id: float(scores[idx]) for node_id, idx in node_index.items()}


def hypergraph_features(
    question_id: str,
    candidates: list[dict[str, Any]],
    question_entities: list[dict[str, Any]],
    passage_entities: dict[str, list[dict[str, Any]]],
    *,
    structure: str = "knowledge_hypergraph",
    iterations: int = 3,
    damping: float = 0.85,
    max_passage_entities: int = 48,
) -> dict[str, dict[str, float]]:
    graph = build_local_hypergraph(
        question_id,
        candidates,
        question_entities,
        passage_entities,
        structure=structure,
        max_passage_entities=max_passage_entities,
    )
    q_entity_ids = _entity_ids(question_entities)
    seed_nodes = [graph.question_node, *(_entity_node(entity_id) for entity_id in q_entity_ids)]
    node_scores = diffuse(graph, seed_nodes, iterations=iterations, damping=damping)

    q_entity_set = set(q_entity_ids)
    edge_type_counts = Counter(edge.edge_type for edge in graph.hyperedges)
    features: dict[str, dict[str, float]] = {}
    for passage_id, p_node in graph.passage_nodes.items():
        p_entity_ids = set(_entity_ids(passage_entities.get(passage_id, []), max_entities=max_passage_entities))
        overlap = q_entity_set & p_entity_ids
        union = q_entity_set | p_entity_ids
        features[passage_id] = {
            "hypergraph_score": node_scores.get(p_node, 0.0),
            "entity_overlap_count": float(len(overlap)),
            "entity_jaccard": len(overlap) / len(union) if union else 0.0,
            "question_entity_coverage": len(overlap) / len(q_entity_set) if q_entity_set else 0.0,
            "passage_entity_count": float(len(p_entity_ids)),
            "local_num_nodes": float(len(graph.nodes)),
            "local_num_hyperedges": float(len(graph.hyperedges)),
            "local_shared_entity_edges": float(edge_type_counts.get("shared_entity_passages", 0)),
            "local_structure_is_pairwise": float(structure == "pairwise_graph"),
            "local_structure_no_knowledge": float(structure == "no_knowledge_hypergraph"),
        }
    return features
