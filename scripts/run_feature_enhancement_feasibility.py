from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_learning_rerank import (  # noqa: E402
    FEATURE_NAMES,
    enriched_feature_vector,
    make_model,
    parse_float_grid,
    predict_probabilities,
    qid_bucket,
)
from src.evaluation.retrieval_metrics import evaluate_retrieval, group_qrels  # noqa: E402
from src.knowledge.entities import candidate_phrases  # noqa: E402
from src.knowledge.normalization import (  # noqa: E402
    abbreviation_set,
    contained_phrase_count,
    entity_variant_set,
    mesh_variant_set,
    normalize_lightweight,
)
from src.rerank.hypergraph import build_feature_rows, entity_map, mesh_map, relations_map  # noqa: E402
from src.utils import load_config, read_jsonl, set_seed, write_json  # noqa: E402


ENHANCED_FEATURE_NAMES = [
    "norm_entity_overlap_count",
    "norm_entity_jaccard",
    "question_norm_entity_coverage",
    "text_phrase_overlap_count",
    "text_phrase_jaccard",
    "abbreviation_match_count",
    "mesh_normalized_overlap_count",
    "mesh_major_overlap_count",
    "mesh_weighted_overlap",
    "question_to_passage_mesh_alias_count",
    "question_to_passage_mesh_alias_coverage",
    "passage_to_question_mesh_alias_count",
    "passage_mesh_major_count",
    "passage_mesh_richness",
    "mesh_weighted_rank_score",
    "norm_entity_rank_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation-only feasibility checks for lightweight normalization and enhanced MeSH features."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-label", default="bioasq")
    parser.add_argument("--predictions", default="outputs/retrieval/hybrid_full_top100.jsonl")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--relations", default=None)
    parser.add_argument("--output-json", default="results/metrics/feature_enhancement_feasibility_bioasq.json")
    parser.add_argument("--output-csv", default="results/tables/feature_enhancement_feasibility_bioasq.csv")
    parser.add_argument("--output-md", default="results/tables/feature_enhancement_feasibility_bioasq.md")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--max-passage-entities", type=int, default=48)
    parser.add_argument("--max-passage-mesh", type=int, default=32)
    parser.add_argument("--split-modulo", type=int, default=5)
    parser.add_argument("--validation-remainders", type=int, nargs="+", default=[3])
    parser.add_argument("--test-remainders", type=int, nargs="+", default=[4])
    parser.add_argument("--max-qids", type=int, default=None, help="Optional deterministic qid limit for smoke tests.")
    parser.add_argument("--model", choices=["logreg", "hist_gradient"], default="hist_gradient")
    parser.add_argument("--c-grid", type=parse_float_grid, default=parse_float_grid("0,0.01,0.1,1.0"))
    parser.add_argument("--blend-grid", type=parse_float_grid, default=parse_float_grid("0,0.1,0.2,0.35,0.5"))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    parser.add_argument("--primary-metric", default="mrr@10")
    return parser.parse_args()


def split_qids(
    qids: list[str],
    *,
    modulo: int,
    validation_remainders: set[int],
    test_remainders: set[int],
) -> dict[str, set[str]]:
    validation = {qid for qid in qids if qid_bucket(qid, modulo) in validation_remainders}
    test = {qid for qid in qids if qid_bucket(qid, modulo) in test_remainders}
    train = set(qids) - validation - test
    if not train or not validation:
        raise ValueError("Train and validation splits must be non-empty.")
    return {"train": train, "validation": validation, "test": test}


def text_map(rows: list[dict[str, Any]], id_key: str, text_key: str) -> dict[str, str]:
    return {str(row[id_key]): str(row.get(text_key, "")) for row in rows}


def phrase_set(text: str, max_ngram: int = 5) -> set[str]:
    candidates = candidate_phrases(text, max_ngram=max_ngram)
    phrases = {normalize_lightweight(canonical) for canonical, _surface in candidates}
    phrases |= {normalize_lightweight(surface) for _canonical, surface in candidates}
    return {phrase for phrase in phrases if phrase}


def mesh_ui_set(rows: list[dict[str, Any]], *, major_only: bool = False) -> set[str]:
    return {
        str(row.get("mesh_ui", ""))
        for row in rows
        if row.get("mesh_ui") and (not major_only or bool(row.get("major_topic", False)))
    }


def overlap_features(left: set[str], right: set[str]) -> tuple[float, float, float]:
    overlap = left & right
    union = left | right
    coverage = len(overlap) / len(left) if left else 0.0
    jaccard = len(overlap) / len(union) if union else 0.0
    return float(len(overlap)), float(jaccard), float(coverage)


def rank_scores(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (-item[1], item[0]))
    scores = [0.0 for _ in values]
    for rank, (idx, value) in enumerate(ordered, start=1):
        scores[idx] = 0.0 if value <= 0 else 1.0 / rank
    return scores


def enhanced_feature_rows(
    features_by_qid: dict[str, list[dict[str, Any]]],
    *,
    question_text: dict[str, str],
    passage_text: dict[str, str],
    question_entities: dict[str, list[dict[str, Any]]],
    passage_entities: dict[str, list[dict[str, Any]]],
    question_mesh: dict[str, list[dict[str, Any]]],
    passage_mesh: dict[str, list[dict[str, Any]]],
) -> dict[tuple[str, str], dict[str, float]]:
    enhanced: dict[tuple[str, str], dict[str, float]] = {}
    entity_variant_cache: dict[str, set[str]] = {}
    mesh_variant_cache: dict[str, set[str]] = {}
    phrase_cache: dict[str, set[str]] = {}
    abbreviation_cache: dict[str, set[str]] = {}
    major_mesh_cache: dict[str, set[str]] = {}

    def cached_entity_variants(record_id: str, rows: list[dict[str, Any]]) -> set[str]:
        if record_id not in entity_variant_cache:
            entity_variant_cache[record_id] = entity_variant_set(rows)
        return entity_variant_cache[record_id]

    def cached_mesh_variants(record_id: str, rows: list[dict[str, Any]]) -> set[str]:
        if record_id not in mesh_variant_cache:
            mesh_variant_cache[record_id] = mesh_variant_set(rows)
        return mesh_variant_cache[record_id]

    def cached_phrases(record_id: str, text: str) -> set[str]:
        if record_id not in phrase_cache:
            phrase_cache[record_id] = phrase_set(text)
        return phrase_cache[record_id]

    def cached_abbreviations(record_id: str, text: str) -> set[str]:
        if record_id not in abbreviation_cache:
            abbreviation_cache[record_id] = abbreviation_set(text)
        return abbreviation_cache[record_id]

    def cached_major_mesh(record_id: str, rows: list[dict[str, Any]]) -> set[str]:
        if record_id not in major_mesh_cache:
            major_mesh_cache[record_id] = mesh_ui_set(rows, major_only=True)
        return major_mesh_cache[record_id]

    for qid, items in features_by_qid.items():
        q_text = question_text.get(qid, "")
        q_entity_variants = cached_entity_variants(f"q:{qid}", question_entities.get(qid, []))
        q_mesh_variants = cached_mesh_variants(f"q:{qid}", question_mesh.get(qid, []))
        q_mesh_ui = mesh_ui_set(question_mesh.get(qid, []))
        q_phrases = cached_phrases(f"q:{qid}", q_text)
        q_abbreviations = cached_abbreviations(f"q:{qid}", q_text)

        per_qid_rows: list[tuple[tuple[str, str], dict[str, float]]] = []
        mesh_weighted_values: list[float] = []
        norm_entity_values: list[float] = []

        for item in items:
            row = item["row"]
            pid = str(row["passage_id"])
            p_text = passage_text.get(pid, "")
            p_entities = passage_entities.get(pid, [])
            p_mesh = passage_mesh.get(pid, [])

            p_entity_variants = cached_entity_variants(f"p:{pid}", p_entities)
            p_mesh_variants = cached_mesh_variants(f"p:{pid}", p_mesh)
            p_major_ui = cached_major_mesh(f"p:{pid}", p_mesh)

            norm_overlap, norm_jaccard, norm_coverage = overlap_features(q_entity_variants, p_entity_variants)
            phrase_overlap = float(contained_phrase_count(p_text, q_phrases))
            phrase_jaccard = phrase_overlap / len(q_phrases) if q_phrases else 0.0
            mesh_norm_overlap, _mesh_norm_jaccard, _mesh_norm_coverage = overlap_features(q_mesh_variants, p_mesh_variants)
            mesh_major_overlap = float(len(q_mesh_ui & p_major_ui))
            q_to_pmesh_alias = float(contained_phrase_count(q_text, p_mesh_variants))
            p_to_qmesh_alias = float(contained_phrase_count(p_text, q_mesh_variants))
            p_mesh_count = len(p_mesh)
            p_major_count = len(p_major_ui)
            abbreviation_match = float(contained_phrase_count(p_text, q_abbreviations))
            mesh_weighted = (
                mesh_norm_overlap
                + 1.5 * mesh_major_overlap
                + 0.5 * q_to_pmesh_alias
                + 0.25 * p_to_qmesh_alias
            )

            values = {
                "norm_entity_overlap_count": norm_overlap,
                "norm_entity_jaccard": norm_jaccard,
                "question_norm_entity_coverage": norm_coverage,
                "text_phrase_overlap_count": phrase_overlap,
                "text_phrase_jaccard": phrase_jaccard,
                "abbreviation_match_count": abbreviation_match,
                "mesh_normalized_overlap_count": mesh_norm_overlap,
                "mesh_major_overlap_count": mesh_major_overlap,
                "mesh_weighted_overlap": mesh_weighted,
                "question_to_passage_mesh_alias_count": q_to_pmesh_alias,
                "question_to_passage_mesh_alias_coverage": q_to_pmesh_alias / len(p_mesh_variants) if p_mesh_variants else 0.0,
                "passage_to_question_mesh_alias_count": p_to_qmesh_alias,
                "passage_mesh_major_count": float(p_major_count),
                "passage_mesh_richness": math.log1p(p_mesh_count),
            }
            key = (qid, pid)
            per_qid_rows.append((key, values))
            mesh_weighted_values.append(mesh_weighted)
            norm_entity_values.append(norm_overlap)

        mesh_rank = rank_scores(mesh_weighted_values)
        entity_rank = rank_scores(norm_entity_values)
        for idx, (key, values) in enumerate(per_qid_rows):
            values["mesh_weighted_rank_score"] = mesh_rank[idx]
            values["norm_entity_rank_score"] = entity_rank[idx]
            enhanced[key] = values

    return enhanced


def build_matrix(
    features_by_qid: dict[str, list[dict[str, Any]]],
    enhanced_features: dict[tuple[str, str], dict[str, float]],
    qids: set[str],
    qrels_by_qid: dict[str, dict[str, float]],
    *,
    top_k: int,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []
    for qid in sorted(qids):
        gold = qrels_by_qid.get(qid, {})
        for item in features_by_qid.get(qid, []):
            passage_id = str(item["row"]["passage_id"])
            vector = enriched_feature_vector(item, top_k)
            vector.update(enhanced_features.get((qid, passage_id), {}))
            rows.append([float(vector.get(name, 0.0)) for name in feature_names])
            labels.append(1 if passage_id in gold else 0)
            meta.append({"qid": qid, "passage_id": passage_id, "item": item, "features": vector})
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64), meta


def minmax_grouped(rows: list[tuple[float, dict[str, Any]]]) -> list[tuple[float, dict[str, Any]]]:
    values = [score for score, _ in rows]
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [(1.0, meta) for _score, meta in rows]
    return [((score - low) / (high - low), meta) for score, meta in rows]


def rerank_validation(
    model: Any,
    features_by_qid: dict[str, list[dict[str, Any]]],
    enhanced_features: dict[tuple[str, str], dict[str, float]],
    qids: set[str],
    *,
    top_k: int,
    feature_names: list[str],
    blend_weight: float,
    retriever_name: str,
) -> list[dict[str, Any]]:
    matrix, _, meta = build_matrix(
        features_by_qid,
        enhanced_features,
        qids,
        {},
        top_k=top_k,
        feature_names=feature_names,
    )
    model_scores = predict_probabilities(model, matrix)
    by_qid_model: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    by_qid_base: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for score, meta_row in zip(model_scores, meta, strict=False):
        qid = meta_row["qid"]
        base_score = float(meta_row["item"]["features"].get("base_rank_score", 0.0))
        by_qid_model[qid].append((float(score), meta_row))
        by_qid_base[qid].append((base_score, meta_row))

    predictions: list[dict[str, Any]] = []
    for qid in sorted(by_qid_model):
        norm_model = minmax_grouped(by_qid_model[qid])
        norm_base = minmax_grouped(by_qid_base[qid])
        base_lookup = {meta_row["passage_id"]: score for score, meta_row in norm_base}
        scored = []
        for model_score, meta_row in norm_model:
            pid = meta_row["passage_id"]
            score = blend_weight * base_lookup[pid] + (1.0 - blend_weight) * model_score
            scored.append((score, meta_row))
        scored.sort(key=lambda pair: (-pair[0], int(pair[1]["item"]["base_rank"]), str(pair[1]["passage_id"])))
        for rank, (score, meta_row) in enumerate(scored[:top_k], start=1):
            row = meta_row["item"]["row"]
            predictions.append(
                {
                    "question_id": row["question_id"],
                    "passage_id": row["passage_id"],
                    "rank": rank,
                    "score": float(score),
                    "retriever": retriever_name,
                    "metadata": {"base_rank": meta_row["item"]["base_rank"]},
                }
            )
    return predictions


def filter_rows_by_qids(rows: list[dict[str, Any]], qids: set[str], top_k: int | None = None) -> list[dict[str, Any]]:
    kept = [row for row in rows if str(row["question_id"]) in qids]
    if top_k is None:
        return kept
    return [row for row in kept if int(row.get("rank", top_k + 1)) <= top_k]


def run_setting(
    *,
    label: str,
    feature_names: list[str],
    args: argparse.Namespace,
    seed: int,
    features_by_qid: dict[str, list[dict[str, Any]]],
    enhanced_features: dict[tuple[str, str], dict[str, float]],
    qrels_by_qid: dict[str, dict[str, float]],
    validation_qrels: list[dict[str, Any]],
    train_qids: set[str],
    validation_qids: set[str],
) -> dict[str, Any]:
    train_x, train_y, _ = build_matrix(
        features_by_qid,
        enhanced_features,
        train_qids,
        qrels_by_qid,
        top_k=args.top_k,
        feature_names=feature_names,
    )
    if train_y.sum() == 0:
        raise ValueError(f"No positive labels in train split for {label}.")

    best: dict[str, Any] | None = None
    trials = []
    for c_value in args.c_grid:
        model = make_model(args.model, c_value, seed)
        model.fit(train_x, train_y)
        for blend_weight in args.blend_grid:
            validation_predictions = rerank_validation(
                model,
                features_by_qid,
                enhanced_features,
                validation_qids,
                top_k=args.top_k,
                feature_names=feature_names,
                blend_weight=blend_weight,
                retriever_name=f"{label}_validation",
            )
            metrics = evaluate_retrieval(validation_qrels, validation_predictions, sorted(set(args.ks)))
            trial = {
                "label": label,
                "c_value": float(c_value),
                "blend_weight": float(blend_weight),
                "mrr@10": float(metrics.get("mrr@10", 0.0)),
                "recall@10": float(metrics.get("recall@10", 0.0)),
                "ndcg@10": float(metrics.get("ndcg@10", 0.0)),
                "metrics": metrics,
            }
            trials.append(trial)
            primary = float(metrics.get(args.primary_metric, 0.0))
            key = (primary, float(metrics.get("mrr@10", 0.0)), float(metrics.get("recall@10", 0.0)), -blend_weight)
            if best is None or key > best["key"]:
                best = {"key": key, "trial": trial}

    assert best is not None
    return {
        "label": label,
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "selected": {key: value for key, value in best["trial"].items() if key != "metrics"},
        "validation_metrics": best["trial"]["metrics"],
        "top_trials": sorted(
            [{key: value for key, value in trial.items() if key != "metrics"} for trial in trials],
            key=lambda row: (-float(row.get(args.primary_metric, row.get("mrr@10", 0.0))), -row["mrr@10"], -row["recall@10"]),
        )[:10],
    }


def feature_separability(
    enhanced_features: dict[tuple[str, str], dict[str, float]],
    features_by_qid: dict[str, list[dict[str, Any]]],
    qrels_by_qid: dict[str, dict[str, float]],
    validation_qids: set[str],
) -> dict[str, dict[str, float]]:
    values: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for qid in validation_qids:
        gold = set(qrels_by_qid.get(qid, {}))
        for item in features_by_qid.get(qid, []):
            pid = str(item["row"]["passage_id"])
            label = 1 if pid in gold else 0
            for name, value in enhanced_features.get((qid, pid), {}).items():
                values[name].append((label, float(value)))

    summary = {}
    for name, rows in sorted(values.items()):
        gold_values = [value for label, value in rows if label == 1]
        non_gold_values = [value for label, value in rows if label == 0]
        summary[name] = {
            "gold_mean": mean(gold_values) if gold_values else 0.0,
            "non_gold_mean": mean(non_gold_values) if non_gold_values else 0.0,
            "gold_positive_rate": mean([1.0 if value > 0 else 0.0 for value in gold_values]) if gold_values else 0.0,
            "non_gold_positive_rate": mean([1.0 if value > 0 else 0.0 for value in non_gold_values]) if non_gold_values else 0.0,
        }
    return summary


def write_summary_tables(
    output_csv: str,
    output_md: str,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_row(label: str, metrics: dict[str, Any], source_metrics: dict[str, Any] | None = None) -> dict[str, str]:
    row = {
        "dataset": "",
        "method": label,
        "mrr@10": f"{float(metrics.get('mrr@10', 0.0)):.4f}",
        "recall@10": f"{float(metrics.get('recall@10', 0.0)):.4f}",
        "ndcg@10": f"{float(metrics.get('ndcg@10', 0.0)):.4f}",
    }
    if source_metrics:
        row["delta_mrr@10"] = f"{float(metrics.get('mrr@10', 0.0)) - float(source_metrics.get('mrr@10', 0.0)):+.4f}"
        row["delta_recall@10"] = f"{float(metrics.get('recall@10', 0.0)) - float(source_metrics.get('recall@10', 0.0)):+.4f}"
        row["delta_ndcg@10"] = f"{float(metrics.get('ndcg@10', 0.0)) - float(source_metrics.get('ndcg@10', 0.0)):+.4f}"
    else:
        row["delta_mrr@10"] = ""
        row["delta_recall@10"] = ""
        row["delta_ndcg@10"] = ""
    return row


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    paths = config["paths"]

    qrels_path = args.qrels or paths.get("qrels", "data/processed/bioasq_qrels.jsonl")
    questions_path = args.questions or paths.get("questions", "data/processed/bioasq_questions.jsonl")
    corpus_path = args.corpus or paths.get("corpus", "data/processed/bioasq_corpus.jsonl")
    question_entities_path = args.question_entities or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    passage_entities_path = args.passage_entities or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")
    question_mesh_path = args.question_mesh or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")
    passage_mesh_path = args.passage_mesh or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")
    relations_path = args.relations or paths.get("primekg_relations", "data/external_knowledge/primekg_project_relations.jsonl")

    predictions = read_jsonl(args.predictions)
    if args.max_qids is not None:
        selected_qids = set(sorted({str(row["question_id"]) for row in predictions})[: args.max_qids])
        predictions = [row for row in predictions if str(row["question_id"]) in selected_qids]
    qrels = read_jsonl(qrels_path)
    qrels_by_qid = group_qrels(qrels)
    question_entities = entity_map(read_jsonl(question_entities_path), "question_id")
    passage_entities = entity_map(read_jsonl(passage_entities_path), "passage_id")
    question_mesh = mesh_map(read_jsonl(question_mesh_path), "question_id") if Path(question_mesh_path).exists() else {}
    passage_mesh = mesh_map(read_jsonl(passage_mesh_path), "passage_id") if Path(passage_mesh_path).exists() else {}
    entity_relations = relations_map(read_jsonl(relations_path)) if Path(relations_path).exists() else {}
    question_text = text_map(read_jsonl(questions_path), "question_id", "question")
    passage_text = text_map(read_jsonl(corpus_path), "passage_id", "text")

    features_by_qid = build_feature_rows(
        predictions,
        question_entities,
        passage_entities,
        question_mesh,
        passage_mesh,
        entity_relations,
        structure="knowledge_hypergraph",
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        iterations=args.iterations,
        damping=args.damping,
        max_passage_entities=args.max_passage_entities,
        max_passage_mesh=args.max_passage_mesh,
    )
    all_qids = sorted(features_by_qid)

    splits = split_qids(
        all_qids,
        modulo=args.split_modulo,
        validation_remainders=set(args.validation_remainders),
        test_remainders=set(args.test_remainders),
    )
    validation_qrels = filter_rows_by_qids(qrels, splits["validation"])
    source_validation_predictions = filter_rows_by_qids(predictions, splits["validation"], args.top_k)
    source_validation_metrics = evaluate_retrieval(validation_qrels, source_validation_predictions, sorted(set(args.ks)))

    enhanced_features = enhanced_feature_rows(
        features_by_qid,
        question_text=question_text,
        passage_text=passage_text,
        question_entities=question_entities,
        passage_entities=passage_entities,
        question_mesh=question_mesh,
        passage_mesh=passage_mesh,
    )
    base_result = run_setting(
        label="base_hgb_features_validation",
        feature_names=FEATURE_NAMES,
        args=args,
        seed=seed,
        features_by_qid=features_by_qid,
        enhanced_features=enhanced_features,
        qrels_by_qid=qrels_by_qid,
        validation_qrels=validation_qrels,
        train_qids=splits["train"],
        validation_qids=splits["validation"],
    )
    enhanced_result = run_setting(
        label="enhanced_norm_mesh_features_validation",
        feature_names=[*FEATURE_NAMES, *ENHANCED_FEATURE_NAMES],
        args=args,
        seed=seed,
        features_by_qid=features_by_qid,
        enhanced_features=enhanced_features,
        qrels_by_qid=qrels_by_qid,
        validation_qrels=validation_qrels,
        train_qids=splits["train"],
        validation_qids=splits["validation"],
    )

    separability = feature_separability(enhanced_features, features_by_qid, qrels_by_qid, splits["validation"])
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset_label": args.dataset_label,
        "validation_only": True,
        "primary_metric": args.primary_metric,
        "paths": {
            "predictions": args.predictions,
            "qrels": qrels_path,
            "questions": questions_path,
            "corpus": corpus_path,
            "question_entities": question_entities_path,
            "passage_entities": passage_entities_path,
            "question_mesh": question_mesh_path,
            "passage_mesh": passage_mesh_path,
        },
        "split": {
            "modulo": args.split_modulo,
            "train_qids": len(splits["train"]),
            "validation_qids": len(splits["validation"]),
            "held_out_test_qids_not_evaluated": len(splits["test"]),
            "validation_remainders": args.validation_remainders,
            "test_remainders": args.test_remainders,
            "max_qids": args.max_qids,
        },
        "source_validation_metrics": source_validation_metrics,
        "base_hgb_features": base_result,
        "enhanced_norm_mesh_features": enhanced_result,
        "enhanced_feature_separability_validation": separability,
    }
    write_json(args.output_json, payload)

    table_rows = [
        metric_row("Hybrid source validation", source_validation_metrics),
        metric_row("Base HGB features validation", base_result["validation_metrics"], source_validation_metrics),
        metric_row("Enhanced norm+MeSH validation", enhanced_result["validation_metrics"], source_validation_metrics),
    ]
    for row in table_rows:
        row["dataset"] = args.dataset_label
    columns = ["dataset", "method", "mrr@10", "recall@10", "ndcg@10", "delta_mrr@10", "delta_recall@10", "delta_ndcg@10"]
    write_summary_tables(args.output_csv, args.output_md, table_rows, columns)

    print(
        {
            "dataset": args.dataset_label,
            "validation_only": True,
            "output_json": args.output_json,
            "output_md": args.output_md,
            "source_mrr@10": source_validation_metrics.get("mrr@10"),
            "base_mrr@10": base_result["validation_metrics"].get("mrr@10"),
            "enhanced_mrr@10": enhanced_result["validation_metrics"].get("mrr@10"),
            "source_recall@10": source_validation_metrics.get("recall@10"),
            "base_recall@10": base_result["validation_metrics"].get("recall@10"),
            "enhanced_recall@10": enhanced_result["validation_metrics"].get("recall@10"),
        }
    )


if __name__ == "__main__":
    main()
