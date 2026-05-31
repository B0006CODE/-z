from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any

from src.knowledge.mesh_hierarchy import MeshDescriptor, descriptor_depth


GENERIC_MESH_NAMES = {
    "adult",
    "aged",
    "animals",
    "child",
    "female",
    "humans",
    "male",
    "middle aged",
    "mice",
    "rats",
    "retrospective studies",
    "prospective studies",
    "risk factors",
    "time factors",
    "treatment outcome",
}

OUTCOME_KEYWORDS = {
    "accuracy",
    "adverse",
    "benefit",
    "biomarker",
    "complication",
    "diagnosis",
    "efficacy",
    "effect",
    "failure",
    "mortality",
    "outcome",
    "phenotype",
    "prevalence",
    "prognosis",
    "progression",
    "recurrence",
    "remission",
    "response",
    "risk",
    "sensitivity",
    "survival",
    "toxicity",
}

EVIDENCE_UNIT_CATEGORIES = ["disease", "intervention", "chemical", "gene", "outcome"]

EVIDENCE_TYPE_PATTERNS: list[tuple[str, float, re.Pattern[str]]] = [
    ("guideline", 1.0, re.compile(r"\b(guideline|practice guideline|recommendation)\b", re.I)),
    ("meta_analysis", 0.95, re.compile(r"\b(meta-analysis|meta analysis|systematic review)\b", re.I)),
    ("randomized_trial", 0.9, re.compile(r"\b(randomi[sz]ed|random allocation|placebo-controlled)\b", re.I)),
    ("clinical_trial", 0.8, re.compile(r"\b(clinical trial|phase [i1]{1,3}\b|phase iv\b)\b", re.I)),
    ("cohort", 0.65, re.compile(r"\b(cohort|prospective|retrospective|case-control)\b", re.I)),
    ("case_report", 0.35, re.compile(r"\b(case report|case series|case study)\b", re.I)),
    ("review", 0.6, re.compile(r"\b(review)\b", re.I)),
]


@dataclass(frozen=True)
class EvidenceUnitConfig:
    max_question_mesh: int = 32
    max_passage_mesh: int = 64
    max_question_entities: int = 64
    max_passage_entities: int = 128
    specific_mesh_min_depth: int = 4
    high_df_ratio_threshold: float = 0.08


def normalize_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _entity_category(entity_type: str, canonical: str) -> str | None:
    raw = entity_type.lower()
    name = canonical.lower()
    if "disease" in raw:
        return "disease"
    if "chemical" in raw:
        return "chemical"
    if "drug" in raw or "therapy" in raw:
        return "intervention"
    if "gene" in raw or "protein" in raw or "pathway" in raw:
        return "gene"
    if any(keyword in name for keyword in OUTCOME_KEYWORDS):
        return "outcome"
    return None


def _mesh_categories(mesh_ui: str, mesh_name: str, hierarchy: dict[str, MeshDescriptor]) -> set[str]:
    categories: set[str] = set()
    name = mesh_name.lower()
    tree_numbers = hierarchy.get(mesh_ui).tree_numbers if mesh_ui in hierarchy else ()
    roots = {tree.split(".", 1)[0] for tree in tree_numbers if tree}
    if "C" in roots:
        categories.add("disease")
    if "D" in roots or "E02" in roots:
        categories.add("intervention")
    if "D" in roots:
        categories.add("chemical")
    if any(keyword in name for keyword in OUTCOME_KEYWORDS):
        categories.add("outcome")
    if "gene" in name or "protein" in name or "receptor" in name:
        categories.add("gene")
    return categories


def _limited(rows: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    return rows[:max_items] if max_items > 0 else rows


def concept_bag(
    *,
    entity_rows: list[dict[str, Any]],
    mesh_rows: list[dict[str, Any]],
    hierarchy: dict[str, MeshDescriptor],
    config: EvidenceUnitConfig,
    is_question: bool,
) -> dict[str, set[str]]:
    max_entities = config.max_question_entities if is_question else config.max_passage_entities
    max_mesh = config.max_question_mesh if is_question else config.max_passage_mesh
    bag: dict[str, set[str]] = {category: set() for category in EVIDENCE_UNIT_CATEGORIES}
    bag["all"] = set()

    for entity in _limited(entity_rows, max_entities):
        canonical = str(entity.get("canonical", "")).strip()
        if not canonical:
            continue
        category = _entity_category(str(entity.get("entity_type", "")), canonical)
        token = "entity:" + str(entity.get("entity_id") or normalize_name(canonical))
        bag["all"].add(token)
        if category:
            bag[category].add(token)
            if category == "chemical":
                bag["intervention"].add(token)

    for term in _limited(mesh_rows, max_mesh):
        mesh_ui = str(term.get("mesh_ui", "")).strip()
        mesh_name = str(term.get("mesh_name", "")).strip()
        if not mesh_ui:
            continue
        token = "mesh:" + mesh_ui
        bag["all"].add(token)
        for category in _mesh_categories(mesh_ui, mesh_name, hierarchy):
            bag[category].add(token)

    return bag


def mesh_df_counts(passage_mesh: dict[str, list[dict[str, Any]]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for rows in passage_mesh.values():
        seen = {str(row.get("mesh_ui", "")).strip() for row in rows if str(row.get("mesh_ui", "")).strip()}
        counts.update(seen)
    return counts


def entity_df_counts(passage_entities: dict[str, list[dict[str, Any]]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for rows in passage_entities.values():
        seen = {
            "entity:" + str(row.get("entity_id") or normalize_name(str(row.get("canonical", ""))))
            for row in rows
            if str(row.get("entity_id") or row.get("canonical", "")).strip()
        }
        counts.update(seen)
    return counts


def evidence_type_score(title: str, text: str, pubmed_record: dict[str, Any] | None = None) -> tuple[float, str]:
    haystack_parts = [title, text]
    if pubmed_record:
        haystack_parts.append(str(pubmed_record.get("title", "")))
        for term in pubmed_record.get("mesh_terms", []):
            haystack_parts.append(str(term.get("descriptor_name", "")))
            for qualifier in term.get("qualifiers", []):
                haystack_parts.append(str(qualifier.get("name", "")))
    haystack = " ".join(part for part in haystack_parts if part)
    best = (0.5, "unspecified")
    for label, score, pattern in EVIDENCE_TYPE_PATTERNS:
        if pattern.search(haystack) and score > best[0]:
            best = (score, label)
    return float(best[0]), best[1]


def _mesh_sets(
    rows: list[dict[str, Any]],
    hierarchy: dict[str, MeshDescriptor],
    mesh_df: Counter[str],
    total_passages: int,
    config: EvidenceUnitConfig,
) -> dict[str, set[str]]:
    all_ids: set[str] = set()
    major: set[str] = set()
    specific: set[str] = set()
    broad: set[str] = set()
    high_df: set[str] = set()
    for row in _limited(rows, config.max_passage_mesh):
        mesh_ui = str(row.get("mesh_ui", "")).strip()
        if not mesh_ui:
            continue
        name = normalize_name(str(row.get("mesh_name", "")))
        all_ids.add(mesh_ui)
        if bool(row.get("major_topic", False)):
            major.add(mesh_ui)
        depth = descriptor_depth(mesh_ui, hierarchy)
        if depth >= config.specific_mesh_min_depth:
            specific.add(mesh_ui)
        if depth and depth <= 2 or name in GENERIC_MESH_NAMES:
            broad.add(mesh_ui)
        if _safe_ratio(mesh_df.get(mesh_ui, 0), total_passages) >= config.high_df_ratio_threshold:
            high_df.add(mesh_ui)
    return {"all": all_ids, "major": major, "specific": specific, "broad": broad, "high_df": high_df}


def _counterfactual_drop(category_scores: dict[str, float], quality: float) -> float:
    if not category_scores:
        return 0.0
    without_best = sum(sorted(category_scores.values(), reverse=True)[1:])
    with_all = sum(category_scores.values())
    if with_all <= 0:
        return 0.0
    return _clip(quality * (1.0 - without_best / with_all))


def candidate_evidence_unit_features(
    *,
    question_id: str,
    passage_id: str,
    base_rank: int,
    row_score: float,
    question_entities: dict[str, list[dict[str, Any]]],
    passage_entities: dict[str, list[dict[str, Any]]],
    question_mesh: dict[str, list[dict[str, Any]]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    mesh_hierarchy: dict[str, MeshDescriptor],
    mesh_df: Counter[str],
    entity_df: Counter[str],
    total_passages: int,
    corpus: dict[str, dict[str, Any]],
    pubmed_metadata: dict[str, dict[str, Any]],
    config: EvidenceUnitConfig,
) -> tuple[dict[str, float], dict[str, Any]]:
    q_entities = question_entities.get(question_id, [])
    p_entities = passage_entities.get(passage_id, [])
    q_mesh_rows = question_mesh.get(question_id, [])
    p_mesh_rows = passage_mesh.get(passage_id, [])
    q_bag = concept_bag(
        entity_rows=q_entities,
        mesh_rows=q_mesh_rows,
        hierarchy=mesh_hierarchy,
        config=config,
        is_question=True,
    )
    p_bag = concept_bag(
        entity_rows=p_entities,
        mesh_rows=p_mesh_rows,
        hierarchy=mesh_hierarchy,
        config=config,
        is_question=False,
    )

    category_overlaps = {category: q_bag[category] & p_bag[category] for category in EVIDENCE_UNIT_CATEGORIES}
    category_scores = {category: float(len(values)) for category, values in category_overlaps.items() if values}
    matched_categories = {category for category, values in category_overlaps.items() if values}
    entity_overlap = q_bag["all"] & p_bag["all"]

    p_mesh_sets = _mesh_sets(p_mesh_rows, mesh_hierarchy, mesh_df, total_passages, config)
    q_mesh_ids = {str(row.get("mesh_ui", "")).strip() for row in q_mesh_rows if str(row.get("mesh_ui", "")).strip()}
    mesh_overlap = q_mesh_ids & p_mesh_sets["all"]
    major_mesh_overlap = q_mesh_ids & p_mesh_sets["major"]
    specific_mesh_overlap = q_mesh_ids & p_mesh_sets["specific"]
    broad_overlap = q_mesh_ids & (p_mesh_sets["broad"] | p_mesh_sets["high_df"])
    useful_mesh_overlap = mesh_overlap - broad_overlap

    corpus_row = corpus.get(passage_id, {})
    metadata = pubmed_metadata.get(passage_id, {})
    evidence_quality_score, evidence_type = evidence_type_score(
        str(corpus_row.get("title", "")),
        str(corpus_row.get("text", "")),
        metadata,
    )
    specificity_values = [descriptor_depth(mesh_ui, mesh_hierarchy) for mesh_ui in useful_mesh_overlap]
    specificity_values = [value for value in specificity_values if value > 0]
    specificity_score = _clip((mean(specificity_values) / 8.0) if specificity_values else 0.0)
    major_score = _clip(len(major_mesh_overlap) / max(len(q_mesh_ids), 1))
    category_coverage = _clip(len(matched_categories) / max(sum(1 for category in EVIDENCE_UNIT_CATEGORIES if q_bag[category]), 1))
    evidence_unit_match = _clip(
        0.30 * _clip(len(useful_mesh_overlap) / max(len(q_mesh_ids - broad_overlap), 1))
        + 0.25 * major_score
        + 0.30 * category_coverage
        + 0.15 * _clip(len(entity_overlap) / max(len(q_bag["all"]), 1))
    )
    broad_penalty = _clip((len(broad_overlap) + len(p_mesh_sets["high_df"] & q_mesh_ids)) / max(len(mesh_overlap), 1))
    hyperedge_quality = _clip(
        0.42 * evidence_unit_match
        + 0.18 * evidence_quality_score
        + 0.22 * specificity_score
        + 0.18 * major_score
        - 0.30 * broad_penalty
    )
    counterfactual_drop = _counterfactual_drop(category_scores, hyperedge_quality)
    rare_entity_matches = {
        token
        for token in entity_overlap
        if _safe_ratio(entity_df.get(token, 0), total_passages) < config.high_df_ratio_threshold
    }
    rank_prior = 1.0 / max(base_rank, 1)

    features = {
        "base_rank_score": rank_prior,
        "candidate_score": float(row_score),
        "evidence_unit_match": evidence_unit_match,
        "evidence_unit_category_count": float(len(matched_categories)),
        "evidence_unit_entity_overlap": float(len(entity_overlap)),
        "evidence_unit_rare_entity_overlap": float(len(rare_entity_matches)),
        "disease_overlap": float(len(category_overlaps["disease"])),
        "intervention_overlap": float(len(category_overlaps["intervention"])),
        "chemical_overlap": float(len(category_overlaps["chemical"])),
        "gene_overlap": float(len(category_overlaps["gene"])),
        "outcome_overlap": float(len(category_overlaps["outcome"])),
        "disease_chemical_gene_category_count": float(
            sum(1 for category in ["disease", "chemical", "gene"] if category_overlaps[category])
        ),
        "major_mesh_overlap": float(len(major_mesh_overlap)),
        "specific_mesh_overlap": float(len(specific_mesh_overlap)),
        "useful_mesh_overlap": float(len(useful_mesh_overlap)),
        "broad_concept_penalty": broad_penalty,
        "mesh_specificity_score": specificity_score,
        "evidence_quality_score": evidence_quality_score,
        "hyperedge_quality": hyperedge_quality,
        "counterfactual_drop": counterfactual_drop,
        "query_concept_coverage": _clip(len(entity_overlap | {"mesh:" + ui for ui in useful_mesh_overlap}) / max(len(q_bag["all"]), 1)),
    }
    details = {
        "evidence_type": evidence_type,
        "matched_categories": sorted(matched_categories),
        "unit_tokens": sorted(
            set().union(*category_overlaps.values(), {"mesh:" + ui for ui in major_mesh_overlap | specific_mesh_overlap})
        ),
        "major_mesh_overlap": sorted(major_mesh_overlap),
        "specific_mesh_overlap": sorted(specific_mesh_overlap),
        "broad_overlap": sorted(broad_overlap),
    }
    return features, details


def add_cluster_features(rows_by_qid: dict[str, list[dict[str, Any]]]) -> None:
    for rows in rows_by_qid.values():
        token_counts: Counter[str] = Counter()
        signature_counts: Counter[tuple[str, ...]] = Counter()
        for item in rows:
            tokens = [token for token in item.get("details", {}).get("unit_tokens", []) if token]
            token_counts.update(set(tokens))
            if tokens:
                signature_counts[tuple(sorted(tokens[:6]))] += 1
        for item in rows:
            tokens = [token for token in item.get("details", {}).get("unit_tokens", []) if token]
            max_token_cluster = max((token_counts[token] for token in tokens), default=0)
            signature = tuple(sorted(tokens[:6]))
            signature_cluster = signature_counts.get(signature, 0) if signature else 0
            cluster_support = math.log1p(max(max_token_cluster, signature_cluster))
            features = item["features"]
            features["unit_token_count"] = float(len(set(tokens)))
            features["cluster_support"] = float(cluster_support)
            features["hyperpath_score"] = float(features.get("hyperedge_quality", 0.0) * cluster_support)
            features["hyperedge_degree"] = float(max_token_cluster)


def build_evidence_unit_rows(
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
    config = config or EvidenceUnitConfig()
    mesh_df = mesh_df_counts(passage_mesh)
    entity_df = entity_df_counts(passage_entities)
    total_passages = max(len(passage_mesh), len(passage_entities), 1)
    output: dict[str, list[dict[str, Any]]] = {}
    for qid, rows in predictions_by_qid.items():
        items: list[dict[str, Any]] = []
        for row in rows:
            pid = str(row["passage_id"])
            base_rank = int(row.get("rank", len(items) + 1))
            features, details = candidate_evidence_unit_features(
                question_id=str(qid),
                passage_id=pid,
                base_rank=base_rank,
                row_score=float(row.get("score", 0.0)),
                question_entities=question_entities,
                passage_entities=passage_entities,
                question_mesh=question_mesh,
                passage_mesh=passage_mesh,
                mesh_hierarchy=mesh_hierarchy,
                mesh_df=mesh_df,
                entity_df=entity_df,
                total_passages=total_passages,
                corpus=corpus,
                pubmed_metadata=pubmed_metadata,
                config=config,
            )
            items.append({"qid": str(qid), "pid": pid, "row": row, "base_rank": base_rank, "features": features, "details": details})
        output[str(qid)] = items
    add_cluster_features(output)
    return output
