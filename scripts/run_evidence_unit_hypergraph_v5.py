from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from lightgbm import LGBMRanker
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval, group_predictions, group_qrels
from src.hypergraph.evidence_unit import EvidenceUnitConfig, build_evidence_unit_rows
from src.knowledge.mesh_hierarchy import load_mesh_hierarchy
from src.rerank.hypergraph import entity_map, mesh_map
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


RETRIEVAL_FEATURES = [
    "base_rank_score",
    "candidate_score",
]

FLAT_EVIDENCE_UNIT_FEATURES = [
    "evidence_unit_match",
    "evidence_unit_category_count",
    "evidence_unit_entity_overlap",
    "evidence_unit_rare_entity_overlap",
    "disease_overlap",
    "intervention_overlap",
    "chemical_overlap",
    "gene_overlap",
    "outcome_overlap",
    "disease_chemical_gene_category_count",
    "major_mesh_overlap",
    "specific_mesh_overlap",
    "useful_mesh_overlap",
    "broad_concept_penalty",
    "mesh_specificity_score",
    "evidence_quality_score",
    "query_concept_coverage",
]

HYPERGRAPH_EVIDENCE_UNIT_FEATURES = [
    "hyperedge_quality",
    "cluster_support",
    "hyperpath_score",
    "hyperedge_degree",
    "unit_token_count",
    "counterfactual_drop",
]

GATE_SCORE_FEATURES = [
    "tail_max_hyperpath",
    "tail_adv_hyperpath",
    "tail_max_hyperedge_quality",
    "tail_adv_quality",
    "tail_adv_query_concept_coverage",
    "top10_broad_concept_penalty",
    "top10_max_evidence_unit_match",
    "hard_evidence_booster_score",
    "hyperpath_weak_top10_score",
    "quality_weak_top10_score",
]

GATE_SCORE_DIRECTIONS = {
    "top10_max_evidence_unit_match": "<=",
}

FEATURE_SETS = {
    "retrieval_ltr": RETRIEVAL_FEATURES,
    "flat_evidence_unit_ltr": [*RETRIEVAL_FEATURES, *FLAT_EVIDENCE_UNIT_FEATURES],
    "evidence_unit_hypergraph_ltr": [
        *RETRIEVAL_FEATURES,
        *FLAT_EVIDENCE_UNIT_FEATURES,
        *HYPERGRAPH_EVIDENCE_UNIT_FEATURES,
    ],
    "without_evidence_quality": [
        *RETRIEVAL_FEATURES,
        *FLAT_EVIDENCE_UNIT_FEATURES,
        *HYPERGRAPH_EVIDENCE_UNIT_FEATURES,
    ],
    "without_major_mesh": [
        *RETRIEVAL_FEATURES,
        *FLAT_EVIDENCE_UNIT_FEATURES,
        *HYPERGRAPH_EVIDENCE_UNIT_FEATURES,
    ],
    "without_cluster_support": [
        *RETRIEVAL_FEATURES,
        *FLAT_EVIDENCE_UNIT_FEATURES,
        *HYPERGRAPH_EVIDENCE_UNIT_FEATURES,
    ],
}
FEATURE_SETS["without_evidence_quality"] = [
    feature for feature in FEATURE_SETS["without_evidence_quality"] if feature != "evidence_quality_score"
]
FEATURE_SETS["without_major_mesh"] = [
    feature for feature in FEATURE_SETS["without_major_mesh"] if feature not in {"major_mesh_overlap", "specific_mesh_overlap"}
]
FEATURE_SETS["without_cluster_support"] = [
    feature for feature in FEATURE_SETS["without_cluster_support"] if feature not in {"cluster_support", "hyperpath_score", "hyperedge_degree"}
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v5 evidence-unit hypergraph diagnostics and a small ranking experiment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictions", default="outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--mesh-hierarchy", default=None)
    parser.add_argument("--pubmed-metadata", default=None)
    parser.add_argument("--use-pubtator-concepts", action="store_true")
    parser.add_argument("--question-pubtator-concepts", default="data/processed/bioasq_question_pubtator_concepts.jsonl")
    parser.add_argument("--passage-pubtator-concepts", default="data/processed/bioasq_passage_pubtator_concepts.jsonl")
    parser.add_argument("--qid-file", default=None)
    parser.add_argument("--max-qids", type=int, default=100)
    parser.add_argument("--hard-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    parser.add_argument("--num-leaves", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=40)
    parser.add_argument("--blend-grid", type=float, nargs="+", default=[0.0, 0.2, 0.5])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--metrics-output", default="results/metrics/evidence_unit_hypergraph_v5_sample100.json")
    parser.add_argument("--table-output", default="results/tables/evidence_unit_hypergraph_v5_sample100.md")
    parser.add_argument("--csv-output", default="results/tables/evidence_unit_hypergraph_v5_sample100.csv")
    parser.add_argument("--separability-output", default="results/tables/evidence_unit_hypergraph_v5_separability_sample100.csv")
    parser.add_argument("--predictions-output", default="outputs/rerank/evidence_unit_hypergraph_v5_sample100.jsonl")
    parser.add_argument("--gated-predictions-output", default="outputs/rerank/evidence_unit_hypergraph_v5_gated_sample100.jsonl")
    parser.add_argument("--gate-max-selected-rate", type=float, default=0.35)
    return parser.parse_args()


def qid_sort_key(qid: str) -> tuple[int, int | str]:
    return (0, int(qid)) if qid.isdigit() else (1, qid)


def load_map(path: str | Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in read_jsonl(path)}


def load_pubmed_metadata(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}
    return {str(row.get("pmid", "")).strip(): row for row in read_jsonl(path) if str(row.get("pmid", "")).strip()}


INVALID_PUBTATOR_CONCEPTS = {
    "",
    "-",
    "a",
    "an",
    "and",
    "disease",
    "human",
    "humans",
    "molecule",
    "molecules",
    "patient",
    "patients",
    "protein",
    "proteins",
    "the",
    "with",
}


def _valid_pubtator_concept(concept: dict[str, Any]) -> bool:
    concept_type = str(concept.get("type", "")).strip().lower()
    if concept_type not in {"disease", "chemical", "gene"}:
        return False
    concept_id = str(concept.get("concept_id", "")).strip()
    name = str(concept.get("name", "")).strip()
    mention = str(concept.get("mention", "")).strip()
    normalized = {value.lower() for value in [concept_id, name, mention] if value}
    if not normalized or normalized & INVALID_PUBTATOR_CONCEPTS:
        return False
    if concept_id == "-":
        return False
    normalized_name = " ".join(re.findall(r"[a-z0-9]+", (name or mention).lower()))
    if normalized_name in INVALID_PUBTATOR_CONCEPTS:
        return False
    if len(normalized_name) <= 1:
        return False
    return True


def load_pubtator_entity_map(path: str | Path | None, key: str) -> dict[str, list[dict[str, Any]]]:
    if not path or not Path(path).exists():
        return {}
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(path):
        row_id = str(row.get(key, "")).strip()
        if not row_id:
            continue
        for concept in row.get("concepts", []):
            if not isinstance(concept, dict) or not _valid_pubtator_concept(concept):
                continue
            concept_type = str(concept.get("type", "")).strip().lower()
            concept_id = str(concept.get("concept_id", "")).strip()
            canonical = str(concept.get("name") or concept.get("mention") or concept_id).strip()
            entity_id = f"pubtator:{concept_type}:{concept_id or canonical.lower()}"
            if entity_id in seen[row_id]:
                continue
            seen[row_id].add(entity_id)
            output[row_id].append(
                {
                    "entity_id": entity_id,
                    "canonical": canonical,
                    "entity_type": concept_type,
                    "source": "pubtator3",
                    "mention": str(concept.get("mention", "")).strip(),
                }
            )
    return dict(output)


def merge_entity_maps(
    base: dict[str, list[dict[str, Any]]],
    extra: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    merged = {key: list(rows) for key, rows in base.items()}
    for key, rows in extra.items():
        existing = {
            str(row.get("entity_id") or row.get("canonical", "")).strip()
            for row in merged.get(key, [])
            if str(row.get("entity_id") or row.get("canonical", "")).strip()
        }
        for row in rows:
            entity_id = str(row.get("entity_id") or row.get("canonical", "")).strip()
            if entity_id and entity_id not in existing:
                merged.setdefault(key, []).append(row)
                existing.add(entity_id)
    return merged


def select_prediction_rows(
    prediction_path: str | Path,
    qrels_by_qid: dict[str, dict[str, float]],
    *,
    top_k: int,
    max_qids: int,
    qid_file: str | None,
    hard_only: bool,
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    requested_qids: set[str] | None = None
    if qid_file:
        requested_qids = {
            line.strip()
            for line in Path(qid_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qid_order: list[str] = []
    seen_qids: set[str] = set()
    with Path(prediction_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = __import__("json").loads(line)
            qid = str(row["question_id"])
            if qid not in qrels_by_qid:
                continue
            if requested_qids is not None and qid not in requested_qids:
                continue
            if qid not in seen_qids:
                seen_qids.add(qid)
                qid_order.append(qid)
            if len(by_qid[qid]) < top_k:
                by_qid[qid].append(row)

    if hard_only:
        hard_qids: list[str] = []
        for qid in qid_order:
            gold = set(qrels_by_qid.get(qid, {}))
            top100 = {str(row["passage_id"]) for row in by_qid.get(qid, [])[:100]}
            top10 = {str(row["passage_id"]) for row in by_qid.get(qid, [])[:10]}
            if gold & top100 and not (gold & top10):
                hard_qids.append(qid)
        qid_order = hard_qids

    qid_order = qid_order[:max_qids]
    qid_set = set(qid_order)
    return qid_order, {qid: by_qid[qid] for qid in qid_order if qid in qid_set}


def flatten_predictions(by_qid: dict[str, list[dict[str, Any]]], qids: set[str], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid in sorted(qids, key=qid_sort_key):
        for rank, row in enumerate(by_qid.get(qid, [])[:top_k], start=1):
            rows.append({**row, "rank": rank})
    return rows


def split_qids(qids: list[str]) -> dict[str, set[str]]:
    if len(qids) < 5:
        raise ValueError("Need at least 5 qids for train/validation/test diagnostics.")
    train_end = max(3, math.ceil(0.6 * len(qids)))
    validation_end = max(train_end + 1, math.ceil(0.8 * len(qids)))
    validation_end = min(validation_end, len(qids) - 1)
    return {
        "train": set(qids[:train_end]),
        "validation": set(qids[train_end:validation_end]),
        "test": set(qids[validation_end:]),
        "all": set(qids),
    }


def feature_matrix(
    feature_rows: dict[str, list[dict[str, Any]]],
    qids: set[str],
    qrels_by_qid: dict[str, dict[str, float]],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int], list[dict[str, Any]]]:
    x_rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    meta: list[dict[str, Any]] = []
    for qid in sorted(qids, key=qid_sort_key):
        items = feature_rows.get(qid, [])
        if not items:
            continue
        groups.append(len(items))
        gold = set(qrels_by_qid.get(qid, {}))
        for item in items:
            pid = str(item["pid"])
            x_rows.append([float(item["features"].get(name, 0.0)) for name in feature_names])
            labels.append(1 if pid in gold else 0)
            meta.append(item)
    return np.asarray(x_rows, dtype=np.float64), np.asarray(labels, dtype=np.int64), groups, meta


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def build_model(seed: int, num_leaves: int, learning_rate: float, n_estimators: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[1, 3, 5, 10],
        random_state=seed,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=5,
        verbose=-1,
    )


def score_model(
    model: LGBMRanker,
    feature_rows: dict[str, list[dict[str, Any]]],
    qids: set[str],
    qrels_by_qid: dict[str, dict[str, float]],
    feature_names: list[str],
    *,
    top_k: int,
    blend_weight: float,
    retriever_name: str,
) -> list[dict[str, Any]]:
    x, _y, _groups, meta = feature_matrix(feature_rows, qids, qrels_by_qid, feature_names)
    scores = model.predict(x) if len(meta) else np.asarray([])
    by_qid: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for score, item in zip(scores, meta, strict=False):
        by_qid[str(item["qid"])].append((float(score), item))

    output: list[dict[str, Any]] = []
    for qid in sorted(by_qid, key=qid_sort_key):
        rows = by_qid[qid]
        model_norm = minmax([score for score, _item in rows])
        base_norm = minmax([float(item["features"].get("base_rank_score", 0.0)) for _score, item in rows])
        scored = []
        for (model_score, item), model_value, base_value in zip(rows, model_norm, base_norm, strict=False):
            final = blend_weight * base_value + (1.0 - blend_weight) * model_value
            scored.append((final, model_score, item))
        scored.sort(key=lambda pair: (-pair[0], int(pair[2]["base_rank"]), str(pair[2]["pid"])))
        for rank, (final, raw_model_score, item) in enumerate(scored[:top_k], start=1):
            row = item["row"]
            output.append(
                {
                    "question_id": row["question_id"],
                    "passage_id": row["passage_id"],
                    "rank": rank,
                    "score": float(final),
                    "retriever": retriever_name,
                    "metadata": {
                        "base_rank": int(item["base_rank"]),
                        "model_score": float(raw_model_score),
                        "blend_weight": float(blend_weight),
                        "features": item["features"],
                        "evidence_unit_details": item["details"],
                    },
                }
            )
    return output


def query_gate_features(feature_rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for qid, rows in feature_rows.items():
        top10 = [item for item in rows if int(item["base_rank"]) <= 10]
        tail = [item for item in rows if int(item["base_rank"]) > 10]
        top10_hyperpath = max((float(item["features"].get("hyperpath_score", 0.0)) for item in top10), default=0.0)
        tail_hyperpath = max((float(item["features"].get("hyperpath_score", 0.0)) for item in tail), default=0.0)
        top10_quality = max((float(item["features"].get("hyperedge_quality", 0.0)) for item in top10), default=0.0)
        tail_quality = max((float(item["features"].get("hyperedge_quality", 0.0)) for item in tail), default=0.0)
        top10_coverage = max((float(item["features"].get("query_concept_coverage", 0.0)) for item in top10), default=0.0)
        tail_coverage = max((float(item["features"].get("query_concept_coverage", 0.0)) for item in tail), default=0.0)
        top10_broad = mean([float(item["features"].get("broad_concept_penalty", 0.0)) for item in top10]) if top10 else 0.0
        top10_match = max((float(item["features"].get("evidence_unit_match", 0.0)) for item in top10), default=0.0)
        features = {
            "tail_max_hyperpath": tail_hyperpath,
            "tail_adv_hyperpath": tail_hyperpath - top10_hyperpath,
            "tail_max_hyperedge_quality": tail_quality,
            "tail_adv_quality": tail_quality - top10_quality,
            "tail_adv_query_concept_coverage": tail_coverage - top10_coverage,
            "top10_broad_concept_penalty": top10_broad,
            "top10_max_evidence_unit_match": top10_match,
        }
        features["hard_evidence_booster_score"] = (
            features["tail_adv_hyperpath"]
            + features["tail_adv_quality"]
            + features["tail_adv_query_concept_coverage"]
            + 0.5 * features["top10_broad_concept_penalty"]
            - features["top10_max_evidence_unit_match"]
        )
        features["hyperpath_weak_top10_score"] = (
            features["tail_adv_hyperpath"] - features["top10_max_evidence_unit_match"]
        )
        features["quality_weak_top10_score"] = (
            features["tail_adv_quality"]
            + features["top10_broad_concept_penalty"]
            - features["top10_max_evidence_unit_match"]
        )
        payload[str(qid)] = {key: float(value) for key, value in features.items()}
    return payload


def _thresholds(values: list[float], direction: str) -> list[float]:
    if not values:
        return [float("inf") if direction == ">=" else float("-inf")]
    unique = sorted(set(float(value) for value in values))
    if direction == "<=":
        return [min(unique) - 1e-9, *unique]
    return [max(unique) + 1e-9, *unique]


def gated_predictions(
    *,
    source_predictions: list[dict[str, Any]],
    intervention_predictions: list[dict[str, Any]],
    gate_features: dict[str, dict[str, float]],
    qids: set[str],
    gate_score_name: str,
    threshold: float,
    gate_direction: str,
    top_k: int,
    retriever_name: str,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    source_by_qid = group_predictions(source_predictions)
    intervention_by_qid = group_predictions(intervention_predictions)
    decisions = {}
    for qid in qids:
        value = float(gate_features.get(str(qid), {}).get(gate_score_name, 0.0))
        decisions[str(qid)] = value <= threshold if gate_direction == "<=" else value >= threshold
    output: list[dict[str, Any]] = []
    for qid in sorted(qids, key=qid_sort_key):
        use_intervention = decisions.get(str(qid), False)
        rows = intervention_by_qid.get(str(qid), []) if use_intervention else source_by_qid.get(str(qid), [])
        for rank, row in enumerate(rows[:top_k], start=1):
            metadata = dict(row.get("metadata", {}))
            metadata["gate"] = {
                "enabled": bool(use_intervention),
                "score_name": gate_score_name,
                "score": float(gate_features.get(str(qid), {}).get(gate_score_name, 0.0)),
                "direction": gate_direction,
                "threshold": float(threshold),
            }
            output.append(
                {
                    **row,
                    "rank": rank,
                    "retriever": retriever_name,
                    "metadata": metadata,
                }
            )
    return output, decisions


def tune_query_gate(
    *,
    source_predictions: list[dict[str, Any]],
    intervention_predictions: list[dict[str, Any]],
    gate_features: dict[str, dict[str, float]],
    validation_qids: set[str],
    validation_qrels: list[dict[str, Any]],
    ks: list[int],
    top_k: int,
    max_selected_rate: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    for score_name in GATE_SCORE_FEATURES:
        direction = GATE_SCORE_DIRECTIONS.get(score_name, ">=")
        values = [float(gate_features.get(qid, {}).get(score_name, 0.0)) for qid in validation_qids]
        for threshold in _thresholds(values, direction):
            predictions, decisions = gated_predictions(
                source_predictions=source_predictions,
                intervention_predictions=intervention_predictions,
                gate_features=gate_features,
                qids=validation_qids,
                gate_score_name=score_name,
                threshold=threshold,
                gate_direction=direction,
                top_k=top_k,
                retriever_name="gated_evidence_unit_hypergraph_validation",
            )
            metrics = evaluate_retrieval(validation_qrels, predictions, sorted(set(ks)))
            selected_count = sum(1 for value in decisions.values() if value)
            trial = {
                "score_name": score_name,
                "direction": direction,
                "threshold": float(threshold),
                "selected_queries": int(selected_count),
                "selected_rate": float(selected_count / max(len(validation_qids), 1)),
                "validation_mrr@10": float(metrics.get("mrr@10", 0.0)),
                "validation_recall@10": float(metrics.get("recall@10", 0.0)),
                "validation_ndcg@10": float(metrics.get("ndcg@10", 0.0)),
            }
            trials.append(trial)
            if trial["selected_rate"] > max_selected_rate:
                continue
            key = (
                trial["validation_ndcg@10"],
                trial["validation_recall@10"],
                trial["validation_mrr@10"],
                -abs(trial["selected_rate"] - 0.25),
            )
            if best is None or key > best["key"]:
                best = {"key": key, "trial": trial}
    assert best is not None
    return {
        "selected": best["trial"],
        "top_trials": sorted(
            trials,
            key=lambda row: (-row["validation_ndcg@10"], -row["validation_recall@10"], -row["validation_mrr@10"]),
        )[:10],
    }


def train_and_eval(
    *,
    label: str,
    feature_names: list[str],
    feature_rows: dict[str, list[dict[str, Any]]],
    splits: dict[str, set[str]],
    qrels: list[dict[str, Any]],
    qrels_by_qid: dict[str, dict[str, float]],
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    train_x, train_y, train_groups, _ = feature_matrix(feature_rows, splits["train"], qrels_by_qid, feature_names)
    if int(train_y.sum()) == 0:
        raise ValueError(f"No positive training labels for {label}.")
    validation_qrels = [row for row in qrels if str(row["question_id"]) in splits["validation"]]
    test_qrels = [row for row in qrels if str(row["question_id"]) in splits["test"]]
    all_qrels = [row for row in qrels if str(row["question_id"]) in splits["all"]]

    model = build_model(seed, args.num_leaves, args.learning_rate, args.n_estimators)
    model.fit(train_x, train_y, group=train_groups)

    trials = []
    best: dict[str, Any] | None = None
    best_validation_predictions: list[dict[str, Any]] = []
    for blend_weight in args.blend_grid:
        validation_predictions = score_model(
            model,
            feature_rows,
            splits["validation"],
            qrels_by_qid,
            feature_names,
            top_k=args.top_k,
            blend_weight=blend_weight,
            retriever_name=f"{label}_validation",
        )
        metrics = evaluate_retrieval(validation_qrels, validation_predictions, sorted(set(args.ks)))
        trial = {
            "blend_weight": float(blend_weight),
            "validation_mrr@10": float(metrics.get("mrr@10", 0.0)),
            "validation_recall@10": float(metrics.get("recall@10", 0.0)),
            "validation_ndcg@10": float(metrics.get("ndcg@10", 0.0)),
        }
        trials.append(trial)
        key = (trial["validation_ndcg@10"], trial["validation_recall@10"], trial["validation_mrr@10"], -blend_weight)
        if best is None or key > best["key"]:
            best = {"key": key, "trial": trial}
            best_validation_predictions = validation_predictions
    assert best is not None

    final_x, final_y, final_groups, _ = feature_matrix(
        feature_rows,
        splits["train"] | splits["validation"],
        qrels_by_qid,
        feature_names,
    )
    final_model = build_model(seed, args.num_leaves, args.learning_rate, args.n_estimators)
    final_model.fit(final_x, final_y, group=final_groups)
    blend = float(best["trial"]["blend_weight"])
    test_predictions = score_model(
        final_model,
        feature_rows,
        splits["test"],
        qrels_by_qid,
        feature_names,
        top_k=args.top_k,
        blend_weight=blend,
        retriever_name=label,
    )
    all_predictions = score_model(
        final_model,
        feature_rows,
        splits["all"],
        qrels_by_qid,
        feature_names,
        top_k=args.top_k,
        blend_weight=blend,
        retriever_name=label,
    )
    diagnostics = {
        "label": label,
        "feature_names": feature_names,
        "selected": best["trial"],
        "top_trials": sorted(trials, key=lambda row: (-row["validation_ndcg@10"], -row["validation_recall@10"]))[:5],
        "test_metrics": evaluate_retrieval(test_qrels, test_predictions, sorted(set(args.ks))),
        "all_sample_metrics": evaluate_retrieval(all_qrels, all_predictions, sorted(set(args.ks))),
        "feature_importance": sorted(
            [
                {"feature": name, "importance": float(value)}
                for name, value in zip(feature_names, final_model.feature_importances_, strict=False)
            ],
            key=lambda row: float(row["importance"]),
            reverse=True,
        ),
        "train_rows": int(train_x.shape[0]),
        "train_positive_rows": int(train_y.sum()),
        "final_train_rows": int(final_x.shape[0]),
        "final_train_positive_rows": int(final_y.sum()),
    }
    return all_predictions, diagnostics, best_validation_predictions


def deterministic_feature_predictions(
    feature_rows: dict[str, list[dict[str, Any]]],
    qids: set[str],
    *,
    top_k: int,
    feature_name: str,
    retriever_name: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for qid in sorted(qids, key=qid_sort_key):
        rows = sorted(
            feature_rows.get(qid, []),
            key=lambda item: (-float(item["features"].get(feature_name, 0.0)), int(item["base_rank"]), str(item["pid"])),
        )
        for rank, item in enumerate(rows[:top_k], start=1):
            row = item["row"]
            output.append(
                {
                    "question_id": row["question_id"],
                    "passage_id": row["passage_id"],
                    "rank": rank,
                    "score": float(item["features"].get(feature_name, 0.0)),
                    "retriever": retriever_name,
                    "metadata": {
                        "base_rank": int(item["base_rank"]),
                        "features": item["features"],
                        "evidence_unit_details": item["details"],
                    },
                }
            )
    return output


def separability(
    feature_rows: dict[str, list[dict[str, Any]]],
    qids: set[str],
    qrels_by_qid: dict[str, dict[str, float]],
    feature_names: list[str],
) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for feature in feature_names:
        labels: list[int] = []
        values: list[float] = []
        for qid in qids:
            gold = set(qrels_by_qid.get(qid, {}))
            for item in feature_rows.get(qid, []):
                labels.append(1 if str(item["pid"]) in gold else 0)
                values.append(float(item["features"].get(feature, 0.0)))
        if not labels or len(set(labels)) < 2:
            auc = 0.0
        else:
            try:
                auc = float(roc_auc_score(labels, values))
            except ValueError:
                auc = 0.0
        gold_values = [value for label, value in zip(labels, values, strict=False) if label == 1]
        non_gold_values = [value for label, value in zip(labels, values, strict=False) if label == 0]
        order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
        top_n = max(1, math.ceil(0.1 * len(order)))
        top_labels = [labels[idx] for idx in order[:top_n]]
        payload[feature] = {
            "auc": auc,
            "gold_mean": mean(gold_values) if gold_values else 0.0,
            "non_gold_mean": mean(non_gold_values) if non_gold_values else 0.0,
            "gold_positive_rate": mean([1.0 if value > 0.0 else 0.0 for value in gold_values]) if gold_values else 0.0,
            "non_gold_positive_rate": mean([1.0 if value > 0.0 else 0.0 for value in non_gold_values]) if non_gold_values else 0.0,
            "top_decile_gold_rate": mean(top_labels) if top_labels else 0.0,
            "base_gold_rate": mean(labels) if labels else 0.0,
            "num_rows": float(len(labels)),
            "num_positive": float(sum(labels)),
        }
    return payload


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_row(method: str, split: str, metrics: dict[str, Any], base: dict[str, Any] | None = None) -> dict[str, str]:
    row = {
        "method": method,
        "split": split,
        "recall@10": f"{float(metrics.get('recall@10', 0.0)):.4f}",
        "mrr@10": f"{float(metrics.get('mrr@10', 0.0)):.4f}",
        "ndcg@10": f"{float(metrics.get('ndcg@10', 0.0)):.4f}",
        "recall@100": f"{float(metrics.get('recall@100', 0.0)):.4f}",
    }
    if base:
        row["delta_recall@10"] = f"{float(metrics.get('recall@10', 0.0)) - float(base.get('recall@10', 0.0)):+.4f}"
        row["delta_mrr@10"] = f"{float(metrics.get('mrr@10', 0.0)) - float(base.get('mrr@10', 0.0)):+.4f}"
        row["delta_ndcg@10"] = f"{float(metrics.get('ndcg@10', 0.0)) - float(base.get('ndcg@10', 0.0)):+.4f}"
    else:
        row["delta_recall@10"] = ""
        row["delta_mrr@10"] = ""
        row["delta_ndcg@10"] = ""
    return row


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(seed)
    paths = config["paths"]

    qrels_path = args.qrels or paths.get("qrels", "data/processed/bioasq_qrels.jsonl")
    questions_path = args.questions or paths.get("questions", "data/processed/bioasq_questions.jsonl")
    corpus_path = args.corpus or paths.get("corpus", "data/processed/bioasq_corpus.jsonl")
    question_entities_path = args.question_entities or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    passage_entities_path = args.passage_entities or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")
    question_mesh_path = args.question_mesh or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")
    passage_mesh_path = args.passage_mesh or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")
    mesh_hierarchy_path = args.mesh_hierarchy or paths.get("mesh_hierarchy", "data/external_knowledge/mesh_hierarchy_2026.jsonl")
    pubmed_metadata_path = args.pubmed_metadata or paths.get("pubmed_mesh", "data/external_knowledge/pubmed_mesh.jsonl")

    qrels = read_jsonl(qrels_path)
    qrels_by_qid = group_qrels(qrels)
    qids, predictions_by_qid = select_prediction_rows(
        args.predictions,
        qrels_by_qid,
        top_k=args.top_k,
        max_qids=args.max_qids,
        qid_file=args.qid_file,
        hard_only=args.hard_only,
    )
    if not qids:
        raise RuntimeError("No qids selected for v5 evidence-unit hypergraph diagnostics.")
    splits = split_qids(qids)
    selected_qids = set(qids)
    selected_qrels = [row for row in qrels if str(row["question_id"]) in selected_qids]
    validation_qrels = [row for row in qrels if str(row["question_id"]) in splits["validation"]]
    test_qrels = [row for row in qrels if str(row["question_id"]) in splits["test"]]
    base_predictions_all = flatten_predictions(predictions_by_qid, selected_qids, args.top_k)
    base_predictions_test = flatten_predictions(predictions_by_qid, splits["test"], args.top_k)
    base_metrics_all = evaluate_retrieval(selected_qrels, base_predictions_all, sorted(set(args.ks)))
    base_metrics_test = evaluate_retrieval(test_qrels, base_predictions_test, sorted(set(args.ks)))

    question_entities = entity_map(read_jsonl(question_entities_path), "question_id")
    passage_entities = entity_map(read_jsonl(passage_entities_path), "passage_id")
    pubtator_stats = {
        "enabled": bool(args.use_pubtator_concepts),
        "question_rows": 0,
        "passage_rows": 0,
        "question_concepts": 0,
        "passage_concepts": 0,
    }
    if args.use_pubtator_concepts:
        question_pubtator = load_pubtator_entity_map(args.question_pubtator_concepts, "question_id")
        passage_pubtator = load_pubtator_entity_map(args.passage_pubtator_concepts, "passage_id")
        pubtator_stats.update(
            {
                "question_rows": len(question_pubtator),
                "passage_rows": len(passage_pubtator),
                "question_concepts": sum(len(rows) for rows in question_pubtator.values()),
                "passage_concepts": sum(len(rows) for rows in passage_pubtator.values()),
            }
        )
        question_entities = merge_entity_maps(question_entities, question_pubtator)
        passage_entities = merge_entity_maps(passage_entities, passage_pubtator)
    question_mesh = mesh_map(read_jsonl(question_mesh_path), "question_id")
    passage_mesh = mesh_map(read_jsonl(passage_mesh_path), "passage_id")
    mesh_hierarchy = load_mesh_hierarchy(read_jsonl(mesh_hierarchy_path))
    corpus = load_map(corpus_path, "passage_id")
    pubmed_metadata = load_pubmed_metadata(pubmed_metadata_path)

    feature_rows = build_evidence_unit_rows(
        predictions_by_qid,
        question_entities=question_entities,
        passage_entities=passage_entities,
        question_mesh=question_mesh,
        passage_mesh=passage_mesh,
        mesh_hierarchy=mesh_hierarchy,
        corpus=corpus,
        pubmed_metadata=pubmed_metadata,
        config=EvidenceUnitConfig(),
    )
    gate_features = query_gate_features(feature_rows)

    separability_features = sorted(set(FLAT_EVIDENCE_UNIT_FEATURES + HYPERGRAPH_EVIDENCE_UNIT_FEATURES))
    sep_all = separability(feature_rows, selected_qids, qrels_by_qid, separability_features)
    sep_test = separability(feature_rows, splits["test"], qrels_by_qid, separability_features)

    method_predictions: dict[str, list[dict[str, Any]]] = {"source_order": base_predictions_all}
    method_diagnostics: dict[str, Any] = {
        "source_order": {"all_sample_metrics": base_metrics_all, "test_metrics": base_metrics_test}
    }
    validation_predictions_by_method: dict[str, list[dict[str, Any]]] = {}
    for label in [
        "retrieval_ltr",
        "flat_evidence_unit_ltr",
        "evidence_unit_hypergraph_ltr",
        "without_evidence_quality",
        "without_major_mesh",
        "without_cluster_support",
    ]:
        predictions, diagnostics, validation_predictions = train_and_eval(
            label=label,
            feature_names=FEATURE_SETS[label],
            feature_rows=feature_rows,
            splits=splits,
            qrels=selected_qrels,
            qrels_by_qid=qrels_by_qid,
            args=args,
            seed=seed,
        )
        method_predictions[label] = predictions
        method_diagnostics[label] = diagnostics
        validation_predictions_by_method[label] = validation_predictions

    validation_source_predictions = flatten_predictions(predictions_by_qid, splits["validation"], args.top_k)
    gate_diagnostics = tune_query_gate(
        source_predictions=validation_source_predictions,
        intervention_predictions=validation_predictions_by_method["evidence_unit_hypergraph_ltr"],
        gate_features=gate_features,
        validation_qids=splits["validation"],
        validation_qrels=validation_qrels,
        ks=args.ks,
        top_k=args.top_k,
        max_selected_rate=args.gate_max_selected_rate,
    )
    selected_gate = gate_diagnostics["selected"]
    gated_predictions_all, gated_decisions_all = gated_predictions(
        source_predictions=base_predictions_all,
        intervention_predictions=method_predictions["evidence_unit_hypergraph_ltr"],
        gate_features=gate_features,
        qids=selected_qids,
        gate_score_name=str(selected_gate["score_name"]),
        threshold=float(selected_gate["threshold"]),
        gate_direction=str(selected_gate.get("direction", ">=")),
        top_k=args.top_k,
        retriever_name="gated_evidence_unit_hypergraph",
    )
    gated_predictions_test = [row for row in gated_predictions_all if str(row["question_id"]) in splits["test"]]
    gated_selected_count = sum(1 for value in gated_decisions_all.values() if value)
    gated_test_selected_count = sum(
        1 for qid, value in gated_decisions_all.items() if qid in splits["test"] and value
    )
    method_predictions["gated_evidence_unit_hypergraph"] = gated_predictions_all
    method_diagnostics["gated_evidence_unit_hypergraph"] = {
        "selected": selected_gate,
        "top_trials": gate_diagnostics["top_trials"],
        "all_sample_metrics": evaluate_retrieval(selected_qrels, gated_predictions_all, sorted(set(args.ks))),
        "test_metrics": evaluate_retrieval(test_qrels, gated_predictions_test, sorted(set(args.ks))),
        "all_selected_queries": int(gated_selected_count),
        "all_selected_rate": float(gated_selected_count / max(len(selected_qids), 1)),
        "test_selected_queries": int(gated_test_selected_count),
        "test_selected_rate": float(gated_test_selected_count / max(len(splits["test"]), 1)),
    }

    hyperpath_predictions = deterministic_feature_predictions(
        feature_rows,
        selected_qids,
        top_k=args.top_k,
        feature_name="hyperpath_score",
        retriever_name="evidence_unit_hyperpath_score_only",
    )
    method_predictions["hyperpath_score_only"] = hyperpath_predictions
    method_diagnostics["hyperpath_score_only"] = {
        "all_sample_metrics": evaluate_retrieval(selected_qrels, hyperpath_predictions, sorted(set(args.ks))),
        "test_metrics": evaluate_retrieval(
            test_qrels,
            [row for row in hyperpath_predictions if str(row["question_id"]) in splits["test"]],
            sorted(set(args.ks)),
        ),
    }

    full_predictions = method_predictions["evidence_unit_hypergraph_ltr"]
    write_jsonl(args.predictions_output, full_predictions)
    write_jsonl(args.gated_predictions_output, method_predictions["gated_evidence_unit_hypergraph"])

    rows = [
        metric_row("Source candidate order", "all_selected", base_metrics_all),
        metric_row("Source candidate order", "held_out_test", base_metrics_test),
    ]
    for label, diagnostics in method_diagnostics.items():
        if label == "source_order":
            continue
        rows.append(metric_row(label, "all_selected", diagnostics.get("all_sample_metrics", {}), base_metrics_all))
        rows.append(metric_row(label, "held_out_test", diagnostics.get("test_metrics", {}), base_metrics_test))
    columns = [
        "method",
        "split",
        "recall@10",
        "mrr@10",
        "ndcg@10",
        "recall@100",
        "delta_recall@10",
        "delta_mrr@10",
        "delta_ndcg@10",
    ]
    write_csv(args.csv_output, rows, columns)
    write_markdown(args.table_output, rows, columns)

    sep_rows = []
    for feature, values in sorted(sep_all.items(), key=lambda item: item[1]["auc"], reverse=True):
        test_values = sep_test.get(feature, {})
        sep_rows.append(
            {
                "feature": feature,
                "auc_all": f"{values.get('auc', 0.0):.4f}",
                "auc_test": f"{test_values.get('auc', 0.0):.4f}",
                "gold_mean": f"{values.get('gold_mean', 0.0):.4f}",
                "non_gold_mean": f"{values.get('non_gold_mean', 0.0):.4f}",
                "top_decile_gold_rate": f"{values.get('top_decile_gold_rate', 0.0):.4f}",
                "base_gold_rate": f"{values.get('base_gold_rate', 0.0):.4f}",
            }
        )
    write_csv(
        args.separability_output,
        sep_rows,
        ["feature", "auc_all", "auc_test", "gold_mean", "non_gold_mean", "top_decile_gold_rate", "base_gold_rate"],
    )

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "method": "evidence-unit hypergraph v5 diagnostics",
        "seed": seed,
        "paths": {
            "predictions": args.predictions,
            "qrels": qrels_path,
            "questions": questions_path,
            "corpus": corpus_path,
            "question_entities": question_entities_path,
            "passage_entities": passage_entities_path,
            "question_mesh": question_mesh_path,
            "passage_mesh": passage_mesh_path,
            "mesh_hierarchy": mesh_hierarchy_path,
            "pubmed_metadata": pubmed_metadata_path,
            "question_pubtator_concepts": args.question_pubtator_concepts,
            "passage_pubtator_concepts": args.passage_pubtator_concepts,
        },
        "pubtator": pubtator_stats,
        "selection": {
            "num_qids": len(qids),
            "qids": qids,
            "hard_only": bool(args.hard_only),
            "max_qids": args.max_qids,
            "top_k": args.top_k,
            "qid_file": args.qid_file,
        },
        "split": {key: sorted(value, key=qid_sort_key) for key, value in splits.items()},
        "base_metrics_all": base_metrics_all,
        "base_metrics_test": base_metrics_test,
        "method_diagnostics": method_diagnostics,
        "gate_features": gate_features,
        "separability_all": sep_all,
        "separability_test": sep_test,
        "outputs": {
            "metrics": args.metrics_output,
            "table": args.table_output,
            "csv": args.csv_output,
            "separability_csv": args.separability_output,
            "predictions": args.predictions_output,
            "gated_predictions": args.gated_predictions_output,
        },
    }
    write_json(args.metrics_output, payload)
    print(
        {
            "metrics": args.metrics_output,
            "table": args.table_output,
            "separability": args.separability_output,
            "source_recall@10": base_metrics_all.get("recall@10"),
            "full_recall@10": method_diagnostics["evidence_unit_hypergraph_ltr"]["all_sample_metrics"].get("recall@10"),
            "source_mrr@10": base_metrics_all.get("mrr@10"),
            "full_mrr@10": method_diagnostics["evidence_unit_hypergraph_ltr"]["all_sample_metrics"].get("mrr@10"),
            "gated_recall@10": method_diagnostics["gated_evidence_unit_hypergraph"]["all_sample_metrics"].get("recall@10"),
            "gated_test_recall@10": method_diagnostics["gated_evidence_unit_hypergraph"]["test_metrics"].get("recall@10"),
            "gate": method_diagnostics["gated_evidence_unit_hypergraph"]["selected"],
            "pubtator": pubtator_stats,
            "best_auc": sep_rows[0] if sep_rows else None,
        }
    )


if __name__ == "__main__":
    main()
