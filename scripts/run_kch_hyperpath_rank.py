from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval, group_qrels
from src.hypergraph.hyperpath import HyperPathConfig, compute_hyperpath_features
from src.knowledge.mesh_hierarchy import load_mesh_hierarchy
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


MAIN_COLUMNS = ["method", "ndcg@10", "mrr@10", "recall@10", "recall@5", "map@10", "precision@10", "delta_ndcg@10", "delta_mrr@10"]
ABLATION_COLUMNS = MAIN_COLUMNS
MAIN_KS = [5, 10, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KCH-HyperPathRank from an existing top-100 reservoir.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictions", default="outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--output-prefix", default="kch_hyperpath")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def qid_bucket(qid: str, modulo: int = 5) -> int:
    try:
        return int(qid) % modulo
    except ValueError:
        return sum(ord(ch) for ch in qid) % modulo


def row_maps(rows: list[dict[str, Any]], id_key: str, value_key: str) -> dict[str, list[dict[str, Any]]]:
    return {str(row[id_key]): list(row.get(value_key, [])) for row in rows}


def relation_adjacency(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source_entity_id", "")).strip()
        target = str(row.get("target_entity_id", "")).strip()
        if not source or not target:
            continue
        relation = {
            "target_entity_id": target,
            "relation": row.get("relation", "related_to"),
            "source": row.get("source", "PrimeKG"),
        }
        reverse = {
            "target_entity_id": source,
            "relation": row.get("relation", "related_to"),
            "source": row.get("source", "PrimeKG"),
        }
        adjacency[source].append(relation)
        adjacency[target].append(reverse)
    return dict(adjacency)


def group_predictions_local(predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["question_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item.get("rank", 10**9)))
    return dict(grouped)


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def filter_predictions(predictions: list[dict[str, Any]], qids: set[str], top_k: int = 100) -> list[dict[str, Any]]:
    return [
        row
        for row in predictions
        if str(row["question_id"]) in qids and int(row.get("rank", top_k + 1)) <= top_k
    ]


def metadata_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = row.get("metadata", {})
    candidates = [metadata] if isinstance(metadata, dict) else []
    for key in ["source_metadata", "base_source_metadata"]:
        nested = metadata.get(key, {}) if isinstance(metadata, dict) else {}
        if isinstance(nested, dict):
            candidates.append(nested)
    return candidates


def source_score(row: dict[str, Any], source: str) -> float:
    for metadata in metadata_candidates(row):
        value = metadata.get("source_scores", {}).get(source)
        if value is not None:
            return float(value)
    return 0.0


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 if value > 0 else 0.0 for value in values]
    return [(value - low) / (high - low) for value in values]


def split_qids(qrels: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    all_qids = sorted({str(row["question_id"]) for row in qrels}, key=lambda qid: (qid_bucket(qid), int(qid) if qid.isdigit() else qid))
    validation = {qid for qid in all_qids if qid_bucket(qid) == 3}
    test = {qid for qid in all_qids if qid_bucket(qid) == 4}
    return validation, test


def shuffled_profiles(
    qids: list[str],
    question_entities: dict[str, list[dict[str, Any]]],
    question_mesh: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    if not qids:
        return {}, {}
    ordered = sorted(qids, key=lambda qid: int(qid) if qid.isdigit() else qid)
    shift = 17 % len(ordered) if len(ordered) > 1 else 0
    entity_map: dict[str, list[dict[str, Any]]] = {}
    mesh_map: dict[str, list[dict[str, Any]]] = {}
    for idx, qid in enumerate(ordered):
        donor = ordered[(idx + shift) % len(ordered)]
        entity_map[qid] = question_entities.get(donor, [])
        mesh_map[qid] = question_mesh.get(donor, [])
    return entity_map, mesh_map


def build_feature_cache(
    qids: set[str],
    preds_by_qid: dict[str, list[dict[str, Any]]],
    question_entities: dict[str, list[dict[str, Any]]],
    passage_entities: dict[str, list[dict[str, Any]]],
    question_mesh: dict[str, list[dict[str, Any]]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    mesh_hierarchy: dict[str, Any],
    relations: dict[str, list[dict[str, Any]]],
    *,
    shuffled: bool = False,
) -> dict[str, dict[str, dict[str, Any]]]:
    if shuffled:
        shuffled_entities, shuffled_mesh = shuffled_profiles(list(qids), question_entities, question_mesh)
    else:
        shuffled_entities, shuffled_mesh = {}, {}
    cfg = HyperPathConfig()
    cache: dict[str, dict[str, dict[str, Any]]] = {}
    for qid in sorted(qids, key=lambda x: int(x) if x.isdigit() else x):
        rows = preds_by_qid.get(qid, [])[:100]
        cache[qid] = compute_hyperpath_features(
            qid,
            rows,
            shuffled_entities.get(qid, question_entities.get(qid, [])),
            passage_entities,
            shuffled_mesh.get(qid, question_mesh.get(qid, [])),
            passage_mesh,
            mesh_hierarchy,
            relations,
            config=cfg,
            include_seed_shared=True,
            include_diffusion=True,
        )
    return cache


def rank_scores_for_query(
    qid: str,
    rows: list[dict[str, Any]],
    feature_rows: dict[str, dict[str, Any]],
    params: dict[str, float],
    *,
    method: str,
) -> list[dict[str, Any]]:
    rows = rows[:100]
    base_values = [1.0 / (60.0 + int(row.get("rank", idx + 1))) for idx, row in enumerate(rows)]
    base_norm = minmax(base_values)
    medcpt_norm = minmax([source_score(row, "medcpt") or source_score(row, "dense") for row in rows])

    flat_raw: list[float] = []
    pairwise_raw: list[float] = []
    for row in rows:
        pid = str(row["passage_id"])
        item = feature_rows.get(pid, {})
        by_type = item.get("path_score_by_type", {})
        flat_raw.append(
            float(by_type.get("exact_mesh", 0.0))
            + float(by_type.get("entity_shared_cluster", 0.0))
            + float(by_type.get("primekg_relation", 0.0))
        )
        pairwise_raw.append(
            float(by_type.get("exact_mesh", 0.0))
            + float(by_type.get("mesh_ancestor_sibling", 0.0))
            + float(by_type.get("entity_shared_cluster", 0.0))
            + float(by_type.get("primekg_relation", 0.0))
        )
    flat_norm = minmax(flat_raw)
    pairwise_norm = minmax(pairwise_raw)

    scored: list[tuple[float, dict[str, Any], dict[str, float], bool, str]] = []
    hp_values = [float(feature_rows.get(str(row["passage_id"]), {}).get("hyperpath_score", 0.0)) for row in rows]
    hp_threshold = sorted(hp_values)[max(0, int(math.floor(0.75 * max(len(hp_values) - 1, 0))))] if hp_values else 0.0
    rescue_quota = int(params.get("rescue_quota", 0.0))
    rescue_candidates: list[tuple[float, int]] = []

    for idx, row in enumerate(rows):
        pid = str(row["passage_id"])
        original_rank = int(row.get("rank", idx + 1))
        item = feature_rows.get(pid, {})
        hp = float(item.get("hyperpath_score", 0.0))
        diff = float(item.get("diffusion_score", 0.0))
        rescue = False
        rescue_mechanism = ""

        if method == "hybrid":
            score = base_norm[idx]
            components = {"base_rank": base_norm[idx]}
        elif method == "flat":
            score = 0.72 * flat_norm[idx] + 0.20 * medcpt_norm[idx] + 0.08 * base_norm[idx]
            components = {"flat_biomedical": flat_norm[idx], "semantic": medcpt_norm[idx], "base_rank": base_norm[idx]}
        elif method == "pairwise":
            score = 0.82 * pairwise_norm[idx] + 0.10 * medcpt_norm[idx] + 0.08 * base_norm[idx]
            components = {"pairwise_path": pairwise_norm[idx], "semantic": medcpt_norm[idx], "base_rank": base_norm[idx]}
        else:
            score = (
                params["hyperpath_weight"] * hp
                + params["diffusion_weight"] * diff
                + params["semantic_weight"] * medcpt_norm[idx]
                + params["base_weight"] * base_norm[idx]
            )
            if original_rank > 10 and hp >= hp_threshold:
                rescue_candidates.append((score, len(scored)))
            components = {
                "hyperpath": params["hyperpath_weight"] * hp,
                "diffusion": params["diffusion_weight"] * diff,
                "semantic": params["semantic_weight"] * medcpt_norm[idx],
                "base_rank": params["base_weight"] * base_norm[idx],
                "rescue_bonus": params["rescue_bonus"] if rescue else 0.0,
            }
        scored.append((score, row, components, rescue, rescue_mechanism))

    rescue_indices: set[int] = set()
    if rescue_quota > 0 and rescue_candidates:
        rescue_candidates.sort(key=lambda item: (-item[0], int(scored[item[1]][1].get("rank", 10**9))))
        rescue_indices = {idx for _, idx in rescue_candidates[:rescue_quota]}
        rescored: list[tuple[float, dict[str, Any], dict[str, float], bool, str]] = []
        for idx, (score, row, components, _, _) in enumerate(scored):
            if idx in rescue_indices:
                components = dict(components)
                components["rescue_bonus"] = params["rescue_bonus"]
                rescored.append((score + params["rescue_bonus"], row, components, True, "rank11_100_high_hyperpath"))
            else:
                rescored.append((score, row, components, False, ""))
        scored = rescored

    if method == "no_expansion":
        top10 = scored[:10]
        rest = scored[10:]
        top10.sort(key=lambda item: (-item[0], int(item[1].get("rank", 10**9)), str(item[1]["passage_id"])))
        scored = top10 + rest
    elif method not in {"hybrid", "flat", "pairwise"}:
        protected_pool = [
            item
            for item in scored
            if int(item[1].get("rank", 10**9)) <= 10 or item[3]
        ]
        protected_ids = {str(item[1]["passage_id"]) for item in protected_pool}
        protected_pool.sort(key=lambda item: (-item[0], int(item[1].get("rank", 10**9)), str(item[1]["passage_id"])))
        top10 = protected_pool[:10]
        top10_ids = {str(item[1]["passage_id"]) for item in top10}
        rest = [
            item
            for item in scored
            if str(item[1]["passage_id"]) not in top10_ids
        ]
        rest.sort(key=lambda item: (int(item[1].get("rank", 10**9)), -item[0], str(item[1]["passage_id"])))
        scored = top10 + rest
    else:
        scored.sort(key=lambda item: (-item[0], int(item[1].get("rank", 10**9)), str(item[1]["passage_id"])))

    predictions: list[dict[str, Any]] = []
    for rank, (score, row, components, rescue, rescue_mechanism) in enumerate(scored, start=1):
        pid = str(row["passage_id"])
        item = feature_rows.get(pid, {})
        predictions.append(
            {
                "question_id": qid,
                "passage_id": pid,
                "rank": rank,
                "score": float(score),
                "retriever": method,
                "metadata": {
                    "hyperpath_score": float(item.get("hyperpath_score", 0.0)),
                    "diffusion_score": float(item.get("diffusion_score", 0.0)),
                    "final_score_components": components,
                    "is_hyperpath_rescued": bool(rescue and rank <= 10),
                    "original_rank": int(row.get("rank", rank)),
                    "rescue_mechanism": rescue_mechanism,
                    "path_score_by_type": item.get("path_score_by_type", {}),
                    "coverage": item.get("coverage", {}),
                    "local_hyperedge_coverage": item.get("local_hyperedge_coverage", {}),
                    "source_metadata": row.get("metadata", {}),
                },
            }
        )
    return predictions


def make_predictions(
    qids: set[str],
    preds_by_qid: dict[str, list[dict[str, Any]]],
    feature_cache: dict[str, dict[str, dict[str, Any]]],
    params: dict[str, float],
    *,
    method: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for qid in sorted(qids, key=lambda x: int(x) if x.isdigit() else x):
        output.extend(rank_scores_for_query(qid, preds_by_qid.get(qid, [])[:100], feature_cache.get(qid, {}), params, method=method))
    return output


def metric_score(metrics: dict[str, Any]) -> float:
    return 0.55 * float(metrics.get("ndcg@10", 0.0)) + 0.30 * float(metrics.get("mrr@10", 0.0)) + 0.15 * float(metrics.get("recall@10", 0.0))


def tune_params(
    validation_qids: set[str],
    qrels: list[dict[str, Any]],
    preds_by_qid: dict[str, list[dict[str, Any]]],
    feature_cache: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    qrels_val = filter_qrels(qrels, validation_qids)
    trials: list[dict[str, Any]] = []
    for hp_w in [0.50, 0.60, 0.70]:
        for diff_w in [0.20, 0.30, 0.40]:
            for sem_w in [0.00, 0.04, 0.08]:
                for base_w in [0.02, 0.06, 0.10]:
                    total = hp_w + diff_w + sem_w + base_w
                    hyper_core = (hp_w + diff_w) / total
                    if hyper_core < 0.82:
                        continue
                    for rescue_quota in [0, 3, 6]:
                        for rescue_bonus in [0.0, 0.04, 0.08]:
                            params = {
                                "hyperpath_weight": hp_w,
                                "diffusion_weight": diff_w,
                                "semantic_weight": sem_w,
                                "base_weight": base_w,
                                "rescue_quota": float(rescue_quota),
                                "rescue_bonus": rescue_bonus,
                            }
                            pred = make_predictions(validation_qids, preds_by_qid, feature_cache, params, method="full")
                            metrics = evaluate_retrieval(qrels_val, pred, MAIN_KS)
                            trials.append({"params": params, "metrics": metrics, "selection_score": metric_score(metrics), "hyper_core_ratio": hyper_core})
    trials.sort(key=lambda row: (-row["selection_score"], -row["metrics"].get("ndcg@10", 0.0), -row["metrics"].get("mrr@10", 0.0)))
    return trials[0]["params"], trials[:10]


def table_row(method: str, metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, str]:
    row = {"method": method}
    for key in ["ndcg@10", "mrr@10", "recall@10", "recall@5", "map@10", "precision@10"]:
        row[key] = f"{float(metrics.get(key, 0.0)):.4f}"
    row["delta_ndcg@10"] = f"{float(metrics.get('ndcg@10', 0.0)) - float(baseline.get('ndcg@10', 0.0)):+.4f}"
    row["delta_mrr@10"] = f"{float(metrics.get('mrr@10', 0.0)) - float(baseline.get('mrr@10', 0.0)):+.4f}"
    return row


def write_table(csv_path: Path, md_path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnostics(
    qids: set[str],
    qrels: list[dict[str, Any]],
    hybrid_predictions: list[dict[str, Any]],
    full_predictions: list[dict[str, Any]],
    feature_cache: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    qrels_by_qid = group_qrels(filter_qrels(qrels, qids))
    hybrid_by_qid = group_predictions_local(hybrid_predictions)
    full_by_qid = group_predictions_local(full_predictions)
    gold_11_100 = 0
    rescued_gold = 0
    positives: list[float] = []
    negatives: list[float] = []
    coverage = Counter()
    for qid in qids:
        gold = set(qrels_by_qid.get(qid, {}))
        hybrid_rows = hybrid_by_qid.get(qid, [])
        rank_by_pid = {str(row["passage_id"]): int(row.get("rank", 10**9)) for row in hybrid_rows}
        if any(11 <= rank_by_pid.get(pid, 10**9) <= 100 for pid in gold):
            gold_11_100 += 1
        for row in full_by_qid.get(qid, [])[:10]:
            metadata = row.get("metadata", {})
            if str(row["passage_id"]) in gold and metadata.get("is_hyperpath_rescued"):
                rescued_gold += 1
        for pid, item in feature_cache.get(qid, {}).items():
            score = float(item.get("hyperpath_score", 0.0))
            if pid in gold:
                positives.append(score)
            else:
                negatives.append(score)
        local = next(iter(feature_cache.get(qid, {}).values()), {}).get("local_hyperedge_coverage", {})
        for key, value in local.items():
            coverage[key] += float(value)
    denom = max(len(qids), 1)
    return {
        "hyperedge_coverage_mean": {key: value / denom for key, value in coverage.items()},
        "queries_with_gold_rank_11_100_reservoir": gold_11_100,
        "rescued_gold_count": rescued_gold,
        "path_score_positive_mean": sum(positives) / len(positives) if positives else 0.0,
        "path_score_negative_mean": sum(negatives) / len(negatives) if negatives else 0.0,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)
    paths = cfg["paths"]

    qrels_path = args.qrels or paths.get("qrels", "data/processed/bioasq_qrels.jsonl")
    qrels = read_jsonl(qrels_path)
    predictions = read_jsonl(args.predictions)
    preds_by_qid = group_predictions_local(predictions)
    validation_qids, test_qids_all = split_qids(qrels)
    test_ordered = sorted(test_qids_all, key=lambda qid: int(qid) if qid.isdigit() else qid)
    eval_qids = set(test_ordered[: args.sample_size]) if args.mode == "sample" else set(test_ordered)

    question_entities = row_maps(read_jsonl(paths["question_entities"]), "question_id", "entities")
    passage_entities = row_maps(read_jsonl(paths["passage_entities"]), "passage_id", "entities")
    question_mesh = row_maps(read_jsonl(paths["question_mesh"]), "question_id", "mesh_terms")
    passage_mesh = row_maps(read_jsonl(paths["passage_mesh"]), "passage_id", "mesh_terms")
    mesh_hierarchy = load_mesh_hierarchy(read_jsonl(paths["mesh_hierarchy"]))
    relations = relation_adjacency(read_jsonl(paths["primekg_relations"]))

    validation_feature_cache = build_feature_cache(
        validation_qids,
        preds_by_qid,
        question_entities,
        passage_entities,
        question_mesh,
        passage_mesh,
        mesh_hierarchy,
        relations,
    )
    best_params, top_trials = tune_params(validation_qids, qrels, preds_by_qid, validation_feature_cache)

    feature_cache = build_feature_cache(
        eval_qids,
        preds_by_qid,
        question_entities,
        passage_entities,
        question_mesh,
        passage_mesh,
        mesh_hierarchy,
        relations,
    )
    shuffled_cache = build_feature_cache(
        eval_qids,
        preds_by_qid,
        question_entities,
        passage_entities,
        question_mesh,
        passage_mesh,
        mesh_hierarchy,
        relations,
        shuffled=True,
    )

    qrels_eval = filter_qrels(qrels, eval_qids)
    method_predictions = {
        "Hybrid RRF": filter_predictions(predictions, eval_qids, 100),
        "Flat biomedical score, no graph": make_predictions(eval_qids, preds_by_qid, feature_cache, best_params, method="flat"),
        "Pairwise graph replacement": make_predictions(eval_qids, preds_by_qid, feature_cache, best_params, method="pairwise"),
        "Shuffled hyperedges": make_predictions(eval_qids, preds_by_qid, shuffled_cache, best_params, method="full"),
        "No expansion": make_predictions(eval_qids, preds_by_qid, feature_cache, best_params, method="no_expansion"),
        "Full KCH-HyperPathRank": make_predictions(eval_qids, preds_by_qid, feature_cache, best_params, method="full"),
    }
    method_metrics = {
        name: evaluate_retrieval(qrels_eval, pred, MAIN_KS)
        for name, pred in method_predictions.items()
    }
    baseline = method_metrics["Hybrid RRF"]
    main_rows = [
        table_row(name, method_metrics[name], baseline)
        for name in ["Hybrid RRF", "Flat biomedical score, no graph", "Pairwise graph replacement", "Full KCH-HyperPathRank"]
    ]
    ablation_rows = [
        table_row(name, method_metrics[name], baseline)
        for name in ["Hybrid RRF", "Flat biomedical score, no graph", "Pairwise graph replacement", "Shuffled hyperedges", "No expansion", "Full KCH-HyperPathRank"]
    ]

    suffix = "sample100" if args.mode == "sample" else "full"
    table_dir = Path("results/tables")
    metrics_dir = Path("results/metrics")
    output_dir = Path("outputs/rerank")
    write_table(table_dir / f"{args.output_prefix}_{suffix}_main.csv", table_dir / f"{args.output_prefix}_{suffix}_main.md", main_rows, MAIN_COLUMNS)
    write_table(table_dir / f"{args.output_prefix}_{suffix}_ablation.csv", table_dir / f"{args.output_prefix}_{suffix}_ablation.md", ablation_rows, ABLATION_COLUMNS)
    full_output = output_dir / f"{args.output_prefix}_{suffix}_full.jsonl"
    write_jsonl(full_output, method_predictions["Full KCH-HyperPathRank"])

    full_metrics = method_metrics["Full KCH-HyperPathRank"]
    diag = diagnostics(eval_qids, qrels, method_predictions["Hybrid RRF"], method_predictions["Full KCH-HyperPathRank"], feature_cache)
    shuffled_drop = float(full_metrics.get("ndcg@10", 0.0)) - float(method_metrics["Shuffled hyperedges"].get("ndcg@10", 0.0))
    pairwise_gain = float(full_metrics.get("ndcg@10", 0.0)) - float(method_metrics["Pairwise graph replacement"].get("ndcg@10", 0.0))
    flat_gain = float(full_metrics.get("ndcg@10", 0.0)) - float(method_metrics["Flat biomedical score, no graph"].get("ndcg@10", 0.0))
    no_expansion_gain = float(full_metrics.get("ndcg@10", 0.0)) - float(method_metrics["No expansion"].get("ndcg@10", 0.0))
    supports_hypergraph = bool(pairwise_gain > 0.01 and flat_gain > 0.01 and shuffled_drop > 0.01 and no_expansion_gain > 0.01)
    write_json(
        metrics_dir / f"{args.output_prefix}_{suffix}_metrics.json",
        {
            "mode": args.mode,
            "sample_size": len(eval_qids),
            "validation_split": "qid % 5 == 3",
            "test_split": "qid % 5 == 4",
            "best_params": best_params,
            "top_validation_trials": top_trials,
            "metrics": method_metrics,
            "diagnostics": diag,
            "supports_hypergraph_as_main_source": supports_hypergraph,
            "support_criteria": {
                "full_minus_pairwise_ndcg@10": pairwise_gain,
                "full_minus_flat_ndcg@10": flat_gain,
                "full_minus_shuffled_ndcg@10": shuffled_drop,
                "full_minus_no_expansion_ndcg@10": no_expansion_gain,
            },
            "outputs": {
                "main_table_csv": str(table_dir / f"{args.output_prefix}_{suffix}_main.csv"),
                "main_table_md": str(table_dir / f"{args.output_prefix}_{suffix}_main.md"),
                "ablation_table_csv": str(table_dir / f"{args.output_prefix}_{suffix}_ablation.csv"),
                "ablation_table_md": str(table_dir / f"{args.output_prefix}_{suffix}_ablation.md"),
                "full_predictions": str(full_output),
            },
        },
    )
    print(json.dumps({"suffix": suffix, "best_params": best_params, "supports_hypergraph": supports_hypergraph, "metrics": method_metrics}, indent=2))


if __name__ == "__main__":
    main()
