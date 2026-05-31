from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from src.hypergraph.evidence_unit import EvidenceUnitConfig, build_evidence_unit_rows
from src.knowledge.mesh_hierarchy import MeshDescriptor, descriptor_tree_numbers


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _mesh_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("mesh_ui", "")).strip() for row in rows if str(row.get("mesh_ui", "")).strip()}


def _topic_tokens(mesh_rows: list[dict[str, Any]], hierarchy: dict[str, MeshDescriptor]) -> set[str]:
    tokens: set[str] = set()
    for mesh_ui in _mesh_ids(mesh_rows):
        tokens.add(f"mesh:{mesh_ui}")
        for tree_number in descriptor_tree_numbers(mesh_ui, hierarchy):
            parts = [part for part in tree_number.split(".") if part]
            if not parts:
                continue
            if len(parts) >= 2:
                tokens.add("mesh_topic:" + ".".join(parts[:2]))
            if len(parts) >= 3:
                tokens.add("mesh_topic3:" + ".".join(parts[:3]))
    return tokens


def _weighted_topic_score(tokens: set[str]) -> float:
    score = 0.0
    for token in tokens:
        if token.startswith("mesh:"):
            score += 1.0
        elif token.startswith("mesh_topic3:"):
            score += 0.55
        elif token.startswith("mesh_topic:"):
            score += 0.25
    return score


def _evidence_tokens(item: dict[str, Any]) -> set[str]:
    return {str(token) for token in item.get("details", {}).get("unit_tokens", []) if str(token).strip()}


def _pico_completeness(features: dict[str, float]) -> float:
    # PICO is approximated without large annotation cost:
    # P= disease, I= intervention/chemical, O= outcome, plus optional gene/biomarker context.
    patient_or_problem = float(features.get("disease_overlap", 0.0)) > 0
    intervention = float(features.get("intervention_overlap", 0.0)) > 0 or float(features.get("chemical_overlap", 0.0)) > 0
    outcome = float(features.get("outcome_overlap", 0.0)) > 0
    context = float(features.get("gene_overlap", 0.0)) > 0
    return (float(patient_or_problem) + float(intervention) + float(outcome) + 0.5 * float(context)) / 3.5


def _add_two_layer_hyperedge_features(
    rows_by_qid: dict[str, list[dict[str, Any]]],
    *,
    question_mesh: dict[str, list[dict[str, Any]]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    mesh_hierarchy: dict[str, MeshDescriptor],
) -> None:
    for qid, rows in rows_by_qid.items():
        q_topics = _topic_tokens(question_mesh.get(qid, []), mesh_hierarchy)
        topic_counts: Counter[str] = Counter()
        evidence_counts: Counter[str] = Counter()
        category_signature_counts: Counter[tuple[str, ...]] = Counter()

        for item in rows:
            pid = str(item["pid"])
            p_topics = _topic_tokens(passage_mesh.get(pid, []), mesh_hierarchy)
            evidence_tokens = _evidence_tokens(item)
            topic_overlap = q_topics & p_topics
            # Topic hyperedges are query anchored. Passage-only topics are kept
            # for inspection but do not create support because they caused broad
            # ungrounded clusters in earlier v6 diagnostics.
            item["details"]["topic_tokens"] = sorted(topic_overlap)
            item["details"]["passage_topic_tokens"] = sorted(p_topics)
            item["details"]["evidence_tokens"] = sorted(evidence_tokens)
            item["details"]["topic_overlap"] = sorted(topic_overlap)
            topic_counts.update(topic_overlap)
            evidence_counts.update(evidence_tokens)
            categories = tuple(sorted(item.get("details", {}).get("matched_categories", [])))
            if categories:
                category_signature_counts[categories] += 1

        top10_topic_union: set[str] = set()
        top10_evidence_union: set[str] = set()
        top10_evidence_quality = 0.0
        for item in rows:
            if int(item.get("base_rank", 999999)) <= 10:
                top10_topic_union.update(item["details"].get("topic_tokens", []))
                top10_evidence_union.update(item["details"].get("evidence_tokens", []))
                top10_evidence_quality = max(
                    top10_evidence_quality,
                    float(item["features"].get("evidence_hyperedge_quality", item["features"].get("hyperedge_quality", 0.0))),
                )

        max_topic_cluster = max(topic_counts.values(), default=1)
        max_evidence_cluster = max(evidence_counts.values(), default=1)
        for item in rows:
            features = item["features"]
            evidence_tokens = set(item["details"].get("evidence_tokens", []))
            topic_tokens = set(item["details"].get("topic_tokens", []))
            topic_overlap = set(item["details"].get("topic_overlap", []))
            matched_categories = tuple(sorted(item.get("details", {}).get("matched_categories", [])))

            evidence_quality = float(features.get("evidence_quality_score", 0.0))
            unit_match = float(features.get("evidence_unit_match", 0.0))
            hyperedge_quality = float(features.get("hyperedge_quality", 0.0))
            pico = _pico_completeness(features)

            evidence_layer_quality = _clip(
                0.42 * hyperedge_quality
                + 0.24 * unit_match
                + 0.18 * evidence_quality
                + 0.16 * pico
                - 0.20 * float(features.get("broad_concept_penalty", 0.0))
            )
            weighted_overlap = _weighted_topic_score(topic_overlap)
            weighted_query = _weighted_topic_score(q_topics)
            topic_layer_support = _clip(
                0.40 * _safe_ratio(max((topic_counts[token] for token in topic_tokens), default=0), max_topic_cluster)
                + 0.45 * _safe_ratio(weighted_overlap, max(weighted_query, 1.0))
                + 0.15 * _clip(math.log1p(len(topic_tokens)) / math.log(12.0))
            )
            evidence_cluster_support = _clip(
                0.60 * _safe_ratio(max((evidence_counts[token] for token in evidence_tokens), default=0), max_evidence_cluster)
                + 0.40 * _safe_ratio(category_signature_counts.get(matched_categories, 0), max(len(rows), 1))
            )

            sufficiency_gain = _clip(
                0.50 * _safe_ratio(len(evidence_tokens - top10_evidence_union), max(len(evidence_tokens), 1))
                + 0.30
                * _safe_ratio(
                    _weighted_topic_score(topic_tokens - top10_topic_union),
                    max(_weighted_topic_score(topic_tokens), 1.0),
                )
                + 0.20 * pico
            )
            unique_evidence = {token for token in evidence_tokens if evidence_counts[token] == 1}
            unique_topic = {token for token in topic_tokens if topic_counts[token] == 1}
            necessity_gain = _clip(
                0.45 * _safe_ratio(len(unique_evidence), max(len(evidence_tokens), 1))
                + 0.35 * _safe_ratio(_weighted_topic_score(unique_topic), max(_weighted_topic_score(topic_tokens), 1.0))
                + 0.20 * float(features.get("counterfactual_drop", 0.0))
            )
            minimal_support_score = _clip(
                0.38 * evidence_layer_quality
                + 0.22 * topic_layer_support
                + 0.20 * sufficiency_gain
                + 0.20 * necessity_gain
            )
            tail = 1.0 if int(item.get("base_rank", 999999)) > 10 else 0.0
            weak_top10 = _clip(1.0 - max(top10_evidence_quality, 0.0))
            hard_rescue_score = _clip(
                tail
                * (
                    0.42 * sufficiency_gain
                    + 0.26 * necessity_gain
                    + 0.20 * evidence_layer_quality
                    + 0.12 * topic_layer_support
                )
                * (0.75 + 0.25 * weak_top10)
            )

            features.update(
                {
                    "evidence_hyperedge_quality": evidence_layer_quality,
                    "topic_hyperedge_support": topic_layer_support,
                    "evidence_topic_bridge_score": _clip(evidence_layer_quality * (0.5 + topic_layer_support)),
                    "topic_cluster_support": topic_layer_support,
                    "evidence_cluster_support": evidence_cluster_support,
                    "pico_completeness": pico,
                    "support_sufficiency_gain": sufficiency_gain,
                    "support_necessity_gain": necessity_gain,
                    "minimal_support_score": minimal_support_score,
                    "hard_rescue_score": hard_rescue_score,
                    "top10_evidence_quality_gap": _clip(evidence_layer_quality - top10_evidence_quality, -1.0, 1.0),
                    "topic_token_count": float(len(topic_tokens)),
                    "evidence_token_count": float(len(evidence_tokens)),
                    "query_topic_coverage": _safe_ratio(weighted_overlap, max(weighted_query, 1.0)),
                }
            )


def build_evidence_unit_v6_rows(
    predictions_by_qid: dict[str, list[dict[str, Any]]],
    *,
    question_entities: dict[str, list[dict[str, Any]]],
    passage_entities: dict[str, list[dict[str, Any]]],
    question_mesh: dict[str, list[dict[str, Any]]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    mesh_hierarchy: dict[str, MeshDescriptor],
    corpus: dict[str, dict[str, Any]],
    pubmed_metadata: dict[str, dict[str, Any]],
    config: EvidenceUnitConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    rows = build_evidence_unit_rows(
        predictions_by_qid,
        question_entities=question_entities,
        passage_entities=passage_entities,
        question_mesh=question_mesh,
        passage_mesh=passage_mesh,
        mesh_hierarchy=mesh_hierarchy,
        corpus=corpus,
        pubmed_metadata=pubmed_metadata,
        config=config or EvidenceUnitConfig(),
    )
    _add_two_layer_hyperedge_features(
        rows,
        question_mesh=question_mesh,
        passage_mesh=passage_mesh,
        mesh_hierarchy=mesh_hierarchy,
    )
    return rows
