from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import dcg, evaluate_retrieval, group_predictions, group_qrels
from src.hypergraph.counterfactual import add_counterfactual_features
from src.hypergraph.precision_hypergraph import (
    PrecisionHypergraphConfig,
    build_precision_features_by_qid,
)
from src.knowledge.mesh_hierarchy import load_mesh_hierarchy
from src.rerank.hypergraph import entity_map, mesh_map, relations_map
from src.rerank.kch_v4_ltr import build_predictions, score_items, train_lambdamart
from src.rerank.selective_gate import GateRule, candidate_gate_rules
from src.rerank.slate_optimizer import SlateWeights, candidate_slate_weights
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def parse_int_grid(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def parse_float_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KCH-MedRank v4 top-10-oriented reranking.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--hybrid-predictions", default="outputs/retrieval/hybrid_full_top100.jsonl")
    parser.add_argument("--semantic-predictions", default=None)
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--mesh-hierarchy", default="data/external_knowledge/mesh_hierarchy_2026.jsonl")
    parser.add_argument("--relations", default=None)
    parser.add_argument("--output-prefix", default="kch_v4")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-qids", type=int, default=None)
    parser.add_argument("--qid-selection-order", choices=["lexical", "numeric"], default="numeric")
    parser.add_argument("--split-modulo", type=int, default=5)
    parser.add_argument("--validation-remainders", type=int, nargs="+", default=[3])
    parser.add_argument("--test-remainders", type=int, nargs="+", default=[4])
    parser.add_argument("--num-leaves-grid", type=parse_int_grid, default=parse_int_grid("7,15"))
    parser.add_argument("--learning-rate-grid", type=parse_float_grid, default=parse_float_grid("0.05"))
    parser.add_argument("--n-estimators-grid", type=parse_int_grid, default=parse_int_grid("40,80"))
    parser.add_argument("--blend-grid", type=parse_float_grid, default=parse_float_grid("0,0.2,0.4"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def qid_bucket(qid: str, modulo: int) -> int:
    if qid.isdigit():
        return int(qid) % modulo
    return sum(ord(char) for char in qid) % modulo


def qid_sort_key(qid: str, order: str) -> tuple[int, int | str]:
    if order == "numeric" and qid.isdigit():
        return (0, int(qid))
    return (1, qid)


def select_qids(predictions: list[dict[str, Any]], args: argparse.Namespace) -> set[str]:
    qids = sorted({str(row["question_id"]) for row in predictions}, key=lambda value: qid_sort_key(value, args.qid_selection_order))
    if args.max_qids is not None:
        qids = qids[: args.max_qids]
    return set(qids)


def split_qids(qids: set[str], args: argparse.Namespace) -> dict[str, set[str]]:
    validation = {qid for qid in qids if qid_bucket(qid, args.split_modulo) in set(args.validation_remainders)}
    test = {qid for qid in qids if qid_bucket(qid, args.split_modulo) in set(args.test_remainders)}
    train = set(qids) - validation - test
    if not train or not validation or not test:
        raise ValueError(
            f"Empty split after qid selection: train={len(train)}, validation={len(validation)}, test={len(test)}."
        )
    return {"train": train, "validation": validation, "test": test}


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def filter_predictions(rows: list[dict[str, Any]], qids: set[str], top_k: int) -> list[dict[str, Any]]:
    grouped = group_predictions(rows)
    output: list[dict[str, Any]] = []
    for qid in sorted(qids, key=lambda value: qid_sort_key(value, "numeric")):
        for rank, row in enumerate(grouped.get(qid, [])[:top_k], start=1):
            output.append({**row, "rank": rank})
    return output


def read_score_predictions(path: str | None) -> dict[tuple[str, str], float]:
    if not path:
        return {}
    return {
        (str(row["question_id"]), str(row["passage_id"])): float(row.get("score", 0.0))
        for row in read_jsonl(path)
    }


def add_evidence_coverage(metrics: dict[str, Any], ks: list[int]) -> dict[str, Any]:
    output = dict(metrics)
    for k in ks:
        if f"recall@{k}" in output:
            output[f"evidence_coverage@{k}"] = output[f"recall@{k}"]
    return output


def semantic_only_predictions(features_by_qid: dict[str, list[dict[str, Any]]], qids: set[str], top_k: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for qid in sorted(qids, key=lambda value: qid_sort_key(value, "numeric")):
        items = features_by_qid.get(qid, [])
        scored = [
            (float(item["features"].get("biomedical_semantic_score", 0.0)), item)
            for item in items
        ]
        scored.sort(key=lambda pair: (-pair[0], int(pair[1]["base_rank"]), str(pair[1]["row"]["passage_id"])))
        for rank, (score, item) in enumerate(scored[:top_k], start=1):
            row = item["row"]
            output.append(
                {
                    "question_id": row["question_id"],
                    "passage_id": row["passage_id"],
                    "rank": rank,
                    "score": float(score),
                    "retriever": "available_biomedical_semantic_reranker",
                    "metadata": {
                        "base_rank": int(item["base_rank"]),
                        "features": item["features"],
                        "source_metadata": row.get("metadata", {}),
                    },
                }
            )
    return output


def per_query_metric(gold: dict[str, float], rows: list[dict[str, Any]], metric: str, k: int) -> float:
    ranked_ids = [str(row["passage_id"]) for row in rows[:k]]
    gold_ids = set(gold)
    hits = gold_ids & set(ranked_ids)
    if metric in {"recall", "evidence_coverage"}:
        return len(hits) / len(gold_ids) if gold_ids else 0.0
    if metric == "hit":
        return 1.0 if hits else 0.0
    if metric == "mrr":
        for rank, pid in enumerate(ranked_ids, start=1):
            if pid in gold_ids:
                return 1.0 / rank
        return 0.0
    if metric == "ndcg":
        gains = [gold.get(pid, 0.0) for pid in ranked_ids]
        ideal = dcg(sorted(gold.values(), reverse=True)[:k])
        return dcg(gains) / ideal if ideal > 0 else 0.0
    if metric == "map":
        running_hits = 0
        precisions: list[float] = []
        for rank, pid in enumerate(ranked_ids, start=1):
            if pid in gold_ids:
                running_hits += 1
                precisions.append(running_hits / rank)
        denominator = min(len(gold_ids), k)
        return sum(precisions) / denominator if denominator else 0.0
    raise ValueError(metric)


def paired_bootstrap(
    qrels: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    candidate_predictions: list[dict[str, Any]],
    *,
    baseline_label: str,
    candidate_label: str,
    num_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    qrels_by_qid = group_qrels(qrels)
    baseline_by_qid = group_predictions(baseline_predictions)
    candidate_by_qid = group_predictions(candidate_predictions)
    qids = sorted(set(qrels_by_qid) & set(baseline_by_qid) & set(candidate_by_qid))
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    for metric in ["ndcg", "mrr", "recall", "map", "hit"]:
        baseline_values = np.asarray([per_query_metric(qrels_by_qid[qid], baseline_by_qid[qid], metric, 10) for qid in qids])
        candidate_values = np.asarray([per_query_metric(qrels_by_qid[qid], candidate_by_qid[qid], metric, 10) for qid in qids])
        delta = candidate_values - baseline_values
        boot = np.empty(num_bootstrap, dtype=np.float64)
        for idx in range(num_bootstrap):
            sample_idx = rng.integers(0, len(delta), size=len(delta))
            boot[idx] = float(np.mean(delta[sample_idx]))
        p_lower = (float(np.sum(boot <= 0.0)) + 1.0) / (num_bootstrap + 1.0)
        p_upper = (float(np.sum(boot >= 0.0)) + 1.0) / (num_bootstrap + 1.0)
        baseline_mean = float(np.mean(baseline_values))
        observed = float(np.mean(delta))
        results.append(
            {
                "metric": metric,
                "k": 10,
                "baseline_mean": baseline_mean,
                "candidate_mean": float(np.mean(candidate_values)),
                "delta": observed,
                "relative_delta_percent": observed / baseline_mean * 100.0 if baseline_mean else 0.0,
                "ci_lower": float(np.quantile(boot, 0.025)),
                "ci_upper": float(np.quantile(boot, 0.975)),
                "p_value_two_sided": min(1.0, 2.0 * min(p_lower, p_upper)),
            }
        )
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "num_paired_questions": len(qids),
        "num_bootstrap": num_bootstrap,
        "seed": seed,
        "results": results,
    }


def select_gate_rule(
    validation_scored: dict[str, list[dict[str, Any]]],
    validation_qrels: list[dict[str, Any]],
    *,
    blend_weight: float,
    top_k: int,
    ks: list[int],
    validation_baseline_mrr: float,
) -> tuple[GateRule, dict[str, Any]]:
    best: tuple[tuple[float, ...], GateRule, dict[str, Any]] | None = None
    trials: list[dict[str, Any]] = []
    for rule in candidate_gate_rules():
        predictions = build_predictions(
            validation_scored,
            retriever_name="kch_v4_gate_validation",
            top_k=top_k,
            blend_weight=blend_weight,
            gate_rule=rule,
        )
        metrics = evaluate_retrieval(validation_qrels, predictions, ks)
        safe_mrr = float(metrics.get("mrr@10", 0.0)) >= validation_baseline_mrr - 0.002
        key = (
            1.0 if safe_mrr else 0.0,
            float(metrics.get("ndcg@10", 0.0)),
            float(metrics.get("mrr@10", 0.0)),
            float(metrics.get("recall@10", 0.0)),
            float(metrics.get("map@10", 0.0)),
        )
        trial = {"rule": asdict(rule), "metrics": metrics, "selection_key": list(key)}
        trials.append(trial)
        if best is None or key > best[0]:
            best = (key, rule, trial)
    if best is None:
        raise ValueError("No gate rule trials completed.")
    return best[1], {"selected": best[2], "top_trials": sorted(trials, key=lambda row: tuple(-float(v) for v in row["selection_key"]))[:10]}


def select_slate_weights(
    validation_scored: dict[str, list[dict[str, Any]]],
    validation_qrels: list[dict[str, Any]],
    *,
    blend_weight: float,
    gate_rule: GateRule | None,
    fixed_intervention_strength: float,
    top_k: int,
    ks: list[int],
) -> tuple[SlateWeights, dict[str, Any]]:
    best: tuple[tuple[float, ...], SlateWeights, dict[str, Any]] | None = None
    disabled_trial: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    for weights in candidate_slate_weights():
        predictions = build_predictions(
            validation_scored,
            retriever_name="kch_v4_slate_validation",
            top_k=top_k,
            blend_weight=blend_weight,
            gate_rule=gate_rule,
            fixed_intervention_strength=fixed_intervention_strength,
            slate_weights=weights,
        )
        metrics = evaluate_retrieval(validation_qrels, predictions, ks)
        key = (
            float(metrics.get("ndcg@10", 0.0)),
            float(metrics.get("mrr@10", 0.0)),
            float(metrics.get("recall@10", 0.0)),
            float(metrics.get("map@10", 0.0)),
            -float(weights.alpha + weights.beta + weights.gamma + weights.delta),
        )
        trial = {"weights": asdict(weights), "metrics": metrics, "selection_key": list(key)}
        if not weights.enabled:
            disabled_trial = trial
        trials.append(trial)
        if best is None or key > best[0]:
            best = (key, weights, trial)
    if best is None:
        raise ValueError("No slate trials completed.")
    if disabled_trial is not None and best[1].enabled:
        disabled_metrics = disabled_trial["metrics"]
        best_metrics = best[2]["metrics"]
        ndcg_gain = float(best_metrics.get("ndcg@10", 0.0)) - float(disabled_metrics.get("ndcg@10", 0.0))
        mrr_gain = float(best_metrics.get("mrr@10", 0.0)) - float(disabled_metrics.get("mrr@10", 0.0))
        if ndcg_gain < 0.005 or mrr_gain < 0.0:
            selected = SlateWeights(enabled=False)
            diagnostics = {
                "selected": disabled_trial,
                "auto_disabled_candidate": best[2],
                "auto_disabled_reason": "validation_gain_below_margin_or_mrr_drop",
                "top_trials": sorted(trials, key=lambda row: tuple(-float(v) for v in row["selection_key"]))[:10],
            }
            return selected, diagnostics
    return best[1], {"selected": best[2], "top_trials": sorted(trials, key=lambda row: tuple(-float(v) for v in row["selection_key"]))[:10]}


def combined_score_lookup(scored_by_qid: dict[str, list[dict[str, Any]]], blend_weight: float) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for qid, rows in scored_by_qid.items():
        for row in rows:
            score = blend_weight * float(row["base_score_norm"]) + (1.0 - blend_weight) * float(row["model_score_norm"])
            lookup[(qid, str(row["pid"]))] = score
    return lookup


def with_fallback_base_scores(
    scored_by_qid: dict[str, list[dict[str, Any]]],
    fallback_lookup: dict[tuple[str, str], float],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for qid, rows in scored_by_qid.items():
        copied_rows: list[dict[str, Any]] = []
        for row in rows:
            copied = dict(row)
            copied["base_score_norm"] = float(fallback_lookup.get((qid, str(row["pid"])), row["base_score_norm"]))
            copied_rows.append(copied)
        output[qid] = copied_rows
    return output


def write_table(path_csv: Path, path_md: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    with path_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    path_md.parent.mkdir(parents=True, exist_ok=True)
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_row(method: str, metrics: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, str]:
    row = {"method": method}
    for key in ["ndcg@10", "mrr@10", "recall@10", "map@10", "hit@10"]:
        row[key] = f"{float(metrics.get(key, 0.0)):.4f}"
    if baseline:
        row["delta_ndcg@10"] = f"{float(metrics.get('ndcg@10', 0.0)) - float(baseline.get('ndcg@10', 0.0)):+.4f}"
        row["delta_mrr@10"] = f"{float(metrics.get('mrr@10', 0.0)) - float(baseline.get('mrr@10', 0.0)):+.4f}"
        row["delta_recall@10"] = f"{float(metrics.get('recall@10', 0.0)) - float(baseline.get('recall@10', 0.0)):+.4f}"
    else:
        row["delta_ndcg@10"] = ""
        row["delta_mrr@10"] = ""
        row["delta_recall@10"] = ""
    return row


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(seed)
    paths = config["paths"]

    qrels_path = args.qrels or paths.get("qrels", "data/processed/bioasq_qrels.jsonl")
    question_entities_path = args.question_entities or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    passage_entities_path = args.passage_entities or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")
    question_mesh_path = args.question_mesh or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")
    passage_mesh_path = args.passage_mesh or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")
    relations_path = args.relations or paths.get("primekg_relations", "data/external_knowledge/primekg_project_relations.jsonl")

    qrels = read_jsonl(qrels_path)
    hybrid_predictions_all = read_jsonl(args.hybrid_predictions)
    selected_qids = select_qids(hybrid_predictions_all, args)
    splits = split_qids(selected_qids, args)
    ks = sorted(set(args.ks))
    semantic_lookup = read_score_predictions(args.semantic_predictions)

    hybrid_predictions = filter_predictions(hybrid_predictions_all, selected_qids, args.top_k)
    predictions_by_qid = group_predictions(hybrid_predictions)
    question_entities = entity_map(read_jsonl(question_entities_path), "question_id")
    passage_entities = entity_map(read_jsonl(passage_entities_path), "passage_id")
    question_mesh = mesh_map(read_jsonl(question_mesh_path), "question_id") if Path(question_mesh_path).exists() else {}
    passage_mesh = mesh_map(read_jsonl(passage_mesh_path), "passage_id") if Path(passage_mesh_path).exists() else {}
    mesh_hierarchy = load_mesh_hierarchy(read_jsonl(args.mesh_hierarchy)) if Path(args.mesh_hierarchy).exists() else {}
    entity_relations = relations_map(read_jsonl(relations_path)) if Path(relations_path).exists() else {}

    print(f"[{datetime.now().isoformat(timespec='seconds')}] Building KCH v4 precision hypergraph features...", flush=True)
    features_by_qid = build_precision_features_by_qid(
        predictions_by_qid,
        question_entities,
        passage_entities,
        question_mesh,
        passage_mesh,
        mesh_hierarchy,
        entity_relations,
        semantic_lookup,
        config=PrecisionHypergraphConfig(top_k=args.top_k),
    )
    add_counterfactual_features(features_by_qid)

    qrels_by_qid = group_qrels(qrels)
    validation_qrels = filter_qrels(qrels, splits["validation"])
    test_qrels = filter_qrels(qrels, splits["test"])
    validation_hybrid = filter_predictions(hybrid_predictions, splits["validation"], args.top_k)
    test_hybrid = filter_predictions(hybrid_predictions, splits["test"], args.top_k)
    validation_hybrid_metrics = add_evidence_coverage(evaluate_retrieval(validation_qrels, validation_hybrid, ks), ks)
    hybrid_metrics = add_evidence_coverage(evaluate_retrieval(test_qrels, test_hybrid, ks), ks)

    semantic_predictions = semantic_only_predictions(features_by_qid, splits["test"], args.top_k)
    semantic_metrics = add_evidence_coverage(evaluate_retrieval(test_qrels, semantic_predictions, ks), ks)

    output_dir = Path("outputs/rerank")
    metrics_dir = Path("results/metrics")
    tables_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / f"{args.output_prefix}_semantic_only_test_top{args.top_k}.jsonl", semantic_predictions)
    write_json(metrics_dir / f"{args.output_prefix}_semantic_only_metrics.json", semantic_metrics)

    method_specs = [
        {
            "display": "Retrieval-only LambdaMART",
            "key": "retrieval_ltr",
            "setting": "retrieval_ltr",
            "gate": False,
            "slate": False,
            "table": "main",
        },
        {
            "display": "Flat biomedical LTR",
            "key": "flat_biomedical_ltr",
            "setting": "flat_biomedical_ltr",
            "gate": False,
            "slate": False,
            "table": "main",
        },
        {
            "display": "Pairwise graph LTR",
            "key": "pairwise_graph_ltr",
            "setting": "pairwise_graph_ltr",
            "gate": False,
            "slate": False,
            "table": "main",
        },
        {
            "display": "KCH-MedRank v4 full",
            "key": "full",
            "setting": "full_kch_v4",
            "gate": True,
            "slate": True,
            "table": "main",
        },
        {
            "display": "Full without selective gate",
            "key": "without_selective_gate",
            "setting": "full_kch_v4",
            "gate": False,
            "slate": True,
            "table": "ablation",
        },
        {
            "display": "Full without counterfactual hyperedge features",
            "key": "without_counterfactual",
            "setting": "without_counterfactual",
            "gate": True,
            "slate": True,
            "table": "ablation",
        },
        {
            "display": "Full without hypergraph slate optimizer",
            "key": "without_slate_optimizer",
            "setting": "full_kch_v4",
            "gate": True,
            "slate": False,
            "table": "ablation",
        },
        {
            "display": "Full without MeSH hierarchy hyperedges",
            "key": "without_mesh_hierarchy",
            "setting": "without_mesh_hierarchy",
            "gate": True,
            "slate": True,
            "table": "ablation",
        },
        {
            "display": "Full without rare entity hyperedges",
            "key": "without_rare_entity",
            "setting": "without_rare_entity",
            "gate": True,
            "slate": True,
            "table": "ablation",
        },
        {
            "display": "Full without shared seed support hyperedges",
            "key": "without_shared_seed",
            "setting": "without_shared_seed",
            "gate": True,
            "slate": True,
            "table": "ablation",
        },
        {
            "display": "Full without semantic-graph agreement",
            "key": "without_semantic_graph_agreement",
            "setting": "without_semantic_graph_agreement",
            "gate": True,
            "slate": True,
            "table": "ablation",
        },
        {
            "display": "Full without broad/high-frequency concept penalties",
            "key": "without_broad_high_df_penalties",
            "setting": "without_broad_high_df_penalties",
            "gate": True,
            "slate": True,
            "table": "ablation",
        },
        {
            "display": "Full without PrimeKG relation features",
            "key": "without_primekg_relation",
            "setting": "without_primekg_relation",
            "gate": True,
            "slate": True,
            "table": "ablation",
        },
    ]

    trained: dict[str, tuple[Any, Any, list[str], dict[str, Any]]] = {}
    method_metrics: dict[str, dict[str, Any]] = {}
    method_predictions: dict[str, list[dict[str, Any]]] = {}
    method_diagnostics: dict[str, Any] = {}
    scored_cache: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}

    for spec in method_specs:
        setting = str(spec["setting"])
        if setting not in trained:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] Training {setting}...", flush=True)
            trained[setting] = train_lambdamart(
                setting,
                features_by_qid,
                splits["train"],
                splits["validation"],
                qrels,
                seed=seed,
                num_leaves_grid=args.num_leaves_grid,
                learning_rate_grid=args.learning_rate_grid,
                n_estimators_grid=args.n_estimators_grid,
                blend_grid=args.blend_grid,
                ks=ks,
            )
        final_model, validation_model, feature_names, diagnostics = trained[setting]
        blend_weight = float(diagnostics["selected"]["blend_weight"])
        validation_cache_key = (setting, "validation")
        if validation_cache_key not in scored_cache:
            scored_cache[validation_cache_key] = score_items(validation_model, features_by_qid, splits["validation"], qrels_by_qid, feature_names)
        validation_scored = scored_cache[validation_cache_key]
        if setting not in {"retrieval_ltr", "flat_biomedical_ltr", "pairwise_graph_ltr"} and "flat_biomedical_ltr" in trained:
            flat_final, flat_validation, flat_features, flat_diagnostics = trained["flat_biomedical_ltr"]
            flat_blend = float(flat_diagnostics["selected"]["blend_weight"])
            flat_validation_key = ("flat_biomedical_ltr", "validation")
            if flat_validation_key not in scored_cache:
                scored_cache[flat_validation_key] = score_items(
                    flat_validation,
                    features_by_qid,
                    splits["validation"],
                    qrels_by_qid,
                    flat_features,
                )
            validation_scored = with_fallback_base_scores(
                validation_scored,
                combined_score_lookup(scored_cache[flat_validation_key], flat_blend),
            )
        gate_rule: GateRule | None = None
        gate_diagnostics: dict[str, Any] = {"enabled": False}
        fixed_intervention = 1.0
        if bool(spec["gate"]):
            gate_rule, gate_diagnostics = select_gate_rule(
                validation_scored,
                validation_qrels,
                blend_weight=blend_weight,
                top_k=args.top_k,
                ks=ks,
                validation_baseline_mrr=float(validation_hybrid_metrics.get("mrr@10", 0.0)),
            )
            gate_diagnostics["enabled"] = True
        slate_weights = SlateWeights(enabled=False)
        slate_diagnostics: dict[str, Any] = {"enabled": False}
        if bool(spec["slate"]):
            slate_weights, slate_diagnostics = select_slate_weights(
                validation_scored,
                validation_qrels,
                blend_weight=blend_weight,
                gate_rule=gate_rule,
                fixed_intervention_strength=fixed_intervention,
                top_k=args.top_k,
                ks=ks,
            )
            slate_diagnostics["enabled"] = slate_weights.enabled

        test_cache_key = (setting, "test")
        if test_cache_key not in scored_cache:
            scored_cache[test_cache_key] = score_items(final_model, features_by_qid, splits["test"], qrels_by_qid, feature_names)
        test_scored = scored_cache[test_cache_key]
        if setting not in {"retrieval_ltr", "flat_biomedical_ltr", "pairwise_graph_ltr"} and "flat_biomedical_ltr" in trained:
            flat_final, _flat_validation, flat_features, flat_diagnostics = trained["flat_biomedical_ltr"]
            flat_blend = float(flat_diagnostics["selected"]["blend_weight"])
            flat_test_key = ("flat_biomedical_ltr", "test")
            if flat_test_key not in scored_cache:
                scored_cache[flat_test_key] = score_items(
                    flat_final,
                    features_by_qid,
                    splits["test"],
                    qrels_by_qid,
                    flat_features,
                )
            test_scored = with_fallback_base_scores(
                test_scored,
                combined_score_lookup(scored_cache[flat_test_key], flat_blend),
            )
        predictions = build_predictions(
            test_scored,
            retriever_name=f"kch_v4_{spec['key']}",
            top_k=args.top_k,
            blend_weight=blend_weight,
            gate_rule=gate_rule,
            fixed_intervention_strength=fixed_intervention,
            slate_weights=slate_weights,
        )
        output_path = output_dir / f"{args.output_prefix}_{spec['key']}_test_top{args.top_k}.jsonl"
        write_jsonl(output_path, predictions)
        metrics = add_evidence_coverage(evaluate_retrieval(test_qrels, predictions, ks), ks)
        metrics["predictions"] = str(output_path)
        metrics["display_name"] = spec["display"]
        write_json(metrics_dir / f"{args.output_prefix}_{spec['key']}_metrics.json", metrics)
        method_metrics[str(spec["key"])] = metrics
        method_predictions[str(spec["key"])] = predictions
        method_diagnostics[str(spec["key"])] = {
            "display_name": spec["display"],
            "setting": setting,
            "feature_names": feature_names,
            "training": diagnostics,
            "gate": gate_diagnostics,
            "slate": slate_diagnostics,
        }
        print(
            {
                "method": spec["key"],
                "ndcg@10": metrics.get("ndcg@10"),
                "mrr@10": metrics.get("mrr@10"),
                "recall@10": metrics.get("recall@10"),
            },
            flush=True,
        )

    bootstrap_specs = [
        ("full_vs_hybrid", "Hybrid RRF", test_hybrid),
        ("full_vs_semantic", "Available biomedical semantic reranker", semantic_predictions),
        ("full_vs_flat", "Flat biomedical LTR", method_predictions["flat_biomedical_ltr"]),
        ("full_vs_pairwise", "Pairwise graph LTR", method_predictions["pairwise_graph_ltr"]),
        ("full_vs_without_selective_gate", "Full without selective gate", method_predictions["without_selective_gate"]),
        (
            "full_vs_without_counterfactual",
            "Full without counterfactual hyperedge features",
            method_predictions["without_counterfactual"],
        ),
        ("full_vs_without_slate", "Full without hypergraph slate optimizer", method_predictions["without_slate_optimizer"]),
    ]
    bootstrap_payloads: dict[str, Any] = {}
    for idx, (name, baseline_label, baseline_rows) in enumerate(bootstrap_specs):
        payload = paired_bootstrap(
            test_qrels,
            baseline_rows,
            method_predictions["full"],
            baseline_label=baseline_label,
            candidate_label="KCH-MedRank v4 full",
            num_bootstrap=args.bootstrap_samples,
            seed=seed + 997 * idx,
        )
        bootstrap_payloads[name] = payload
        write_json(metrics_dir / f"{args.output_prefix}_{name}_bootstrap.json", payload)

    main_rows = [
        table_row("Hybrid RRF", hybrid_metrics, hybrid_metrics),
        table_row("Available biomedical semantic reranker", semantic_metrics, hybrid_metrics),
        table_row("Retrieval-only LambdaMART", method_metrics["retrieval_ltr"], hybrid_metrics),
        table_row("Flat biomedical LTR", method_metrics["flat_biomedical_ltr"], hybrid_metrics),
        table_row("Pairwise graph LTR", method_metrics["pairwise_graph_ltr"], hybrid_metrics),
        table_row("KCH-MedRank v4 full", method_metrics["full"], hybrid_metrics),
    ]
    ablation_rows = [
        table_row("KCH-MedRank v4 full", method_metrics["full"], method_metrics["full"]),
        table_row("Full without selective gate", method_metrics["without_selective_gate"], method_metrics["full"]),
        table_row("Full without counterfactual hyperedge features", method_metrics["without_counterfactual"], method_metrics["full"]),
        table_row("Full without hypergraph slate optimizer", method_metrics["without_slate_optimizer"], method_metrics["full"]),
        table_row("Full without MeSH hierarchy hyperedges", method_metrics["without_mesh_hierarchy"], method_metrics["full"]),
        table_row("Full without rare entity hyperedges", method_metrics["without_rare_entity"], method_metrics["full"]),
        table_row("Full without shared seed support hyperedges", method_metrics["without_shared_seed"], method_metrics["full"]),
        table_row("Full without semantic-graph agreement", method_metrics["without_semantic_graph_agreement"], method_metrics["full"]),
        table_row("Full without broad/high-frequency concept penalties", method_metrics["without_broad_high_df_penalties"], method_metrics["full"]),
        table_row("Full without PrimeKG relation features", method_metrics["without_primekg_relation"], method_metrics["full"]),
    ]
    columns = [
        "method",
        "ndcg@10",
        "mrr@10",
        "recall@10",
        "map@10",
        "hit@10",
        "delta_ndcg@10",
        "delta_mrr@10",
        "delta_recall@10",
    ]
    write_table(
        tables_dir / f"{args.output_prefix}_main_top10.csv",
        tables_dir / f"{args.output_prefix}_main_top10.md",
        main_rows,
        columns,
    )
    write_table(
        tables_dir / f"{args.output_prefix}_strict_ablation.csv",
        tables_dir / f"{args.output_prefix}_strict_ablation.md",
        ablation_rows,
        columns,
    )

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "method": "KCH-MedRank v4: Selective Counterfactual Hypergraph Reranking",
        "qrels": qrels_path,
        "hybrid_predictions": args.hybrid_predictions,
        "semantic_predictions": args.semantic_predictions,
        "semantic_source": args.semantic_predictions or "hybrid metadata dense/medcpt fallback",
        "split": {
            "modulo": args.split_modulo,
            "validation_remainders": args.validation_remainders,
            "test_remainders": args.test_remainders,
            "max_qids": args.max_qids,
            "train_qids": len(splits["train"]),
            "validation_qids": len(splits["validation"]),
            "test_qids": len(splits["test"]),
        },
        "top_k": args.top_k,
        "ks": ks,
        "baseline_metrics": {
            "Hybrid RRF": hybrid_metrics,
            "Available biomedical semantic reranker": semantic_metrics,
        },
        "method_metrics": method_metrics,
        "diagnostics": method_diagnostics,
        "bootstrap": bootstrap_payloads,
        "outputs": {
            "main_table": str(tables_dir / f"{args.output_prefix}_main_top10.md"),
            "ablation_table": str(tables_dir / f"{args.output_prefix}_strict_ablation.md"),
            "full_predictions": str(output_dir / f"{args.output_prefix}_full_test_top{args.top_k}.jsonl"),
        },
    }
    write_json(metrics_dir / f"{args.output_prefix}_metrics.json", payload)
    write_json(Path(paths.get("logs_dir", "logs")) / f"run_{args.output_prefix}_summary.json", payload)
    print(
        {
            "metrics": str(metrics_dir / f"{args.output_prefix}_metrics.json"),
            "main_table": str(tables_dir / f"{args.output_prefix}_main_top10.md"),
            "ablation_table": str(tables_dir / f"{args.output_prefix}_strict_ablation.md"),
            "hybrid_ndcg@10": hybrid_metrics.get("ndcg@10"),
            "full_ndcg@10": method_metrics["full"].get("ndcg@10"),
            "full_mrr@10": method_metrics["full"].get("mrr@10"),
            "full_recall@10": method_metrics["full"].get("recall@10"),
        }
    )


if __name__ == "__main__":
    main()
