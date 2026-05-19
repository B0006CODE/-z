from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.knowledge.entities import normalize_text


def entity_name_index(dictionary_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in dictionary_rows:
        entity_id = str(row.get("entity_id", ""))
        canonical = str(row.get("canonical", ""))
        entity_type = str(row.get("entity_type", "biomedical_concept"))
        names = [canonical, *[str(item) for item in row.get("surface_forms", [])]]
        for name in names:
            normalized = normalize_text(name)
            if not normalized:
                continue
            key = (normalized, entity_id)
            if key in seen:
                continue
            seen.add(key)
            index[normalized].append(
                {
                    "entity_id": entity_id,
                    "canonical": canonical,
                    "entity_type": entity_type,
                }
            )
    return dict(index)


def relation_adjacency(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        source_id = str(row.get("source_entity_id", ""))
        target_id = str(row.get("target_entity_id", ""))
        relation = str(row.get("relation", "related_to"))
        if not source_id or not target_id:
            continue
        key = (source_id, target_id, relation)
        reverse_key = (target_id, source_id, relation)
        if key in seen:
            continue
        seen.add(key)
        seen.add(reverse_key)
        payload = {
            "relation": relation,
            "display_relation": str(row.get("display_relation", relation)),
            "source_entity_id": source_id,
            "target_entity_id": target_id,
            "source_name": str(row.get("source_name", "")),
            "target_name": str(row.get("target_name", "")),
            "source_type": str(row.get("source_type", "")),
            "target_type": str(row.get("target_type", "")),
        }
        adjacency[source_id].append(payload)
        adjacency[target_id].append({**payload, "source_entity_id": target_id, "target_entity_id": source_id})
    return dict(adjacency)
