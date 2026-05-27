from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import dcg, evaluate_retrieval, group_predictions, group_qrels
from src.utils import read_jsonl, write_json, write_jsonl


def parse_labeled_path(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH.")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=PATH.")
    return label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze effective hard subsets and learned query-level gates.")
    parser.add_argument("--qrels", default="data/processed/bioasq_qrels.jsonl")
    parser.add_argument("--reference-hybrid", default="outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl")
    parser.add_argument("--expanded", default="outputs/retrieval/concept_hg_shared_clusters_full_top300.jsonl")
    parser.add_argument("--method", action="append", type=parse_labeled_path, required=True)
    parser.add_argument("--validation-method", action="append", type=parse_labeled_path, default=[])
    parser.add_argument("--output", default="outputs/rerank/hard_subset_learned_gate_test_top300.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/hard_subset_learned_gate_metrics.json")
    parser.add_argument("--table-output", default="results/tables/hard_subset_learned_gate.md")
    parser.add_argument("--gate-table-output", default="results/tables/query_level_gate_analysis.md")
    parser.add_argument("--primary", choices=["recall@10", "mrr@10", "ndcg@10"], default="recall@10")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 100, 200, 300])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank <= 10:
        return "1-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    if rank <= 100:
        return "51-100"
    if rank <= 200:
        return "101-200"
    if rank <= 300:
        return "201-300"
    return ">300"


def min_gold_rank(rows: list[dict[str, Any]], gold: set[str], limit: int) -> int | None:
    for row in rows[:limit]:
        if str(row["passage_id"]) in gold:
            return int(row["rank"])
    return None


def hard_subset(
    qrels_by_qid: dict[str, dict[str, float]],
    reference_by_qid: dict[str, list[dict[str, Any]]],
    expanded_by_qid: dict[str, list[dict[str, Any]]],
    qids: set[str],
) -> tuple[set[str], dict[str, Any]]:
    hard: set[str] = set()
    hybrid_rank_buckets: Counter[str] = Counter()
    expanded_rank_buckets: Counter[str] = Counter()
    hard_source_counts: Counter[str] = Counter()
    gold_count = 0
    for qid in sorted(qids):
        gold = set(qrels_by_qid.get(qid, {}))
        if not gold:
            continue
        reference_rows = reference_by_qid.get(qid, [])
        expanded_rows = expanded_by_qid.get(qid, [])
        ref_top10_rank = min_gold_rank(reference_rows, gold, 10)
        ref_top100_rank = min_gold_rank(reference_rows, gold, 100)
        exp_top300_rank = min_gold_rank(expanded_rows, gold, 300)
        if ref_top10_rank is not None:
            continue
        if ref_top100_rank is None and exp_top300_rank is None:
            continue
        hard.add(qid)
        gold_count += len(gold)
        hybrid_rank_buckets[rank_bucket(ref_top100_rank)] += 1
        expanded_rank_buckets[rank_bucket(exp_top300_rank)] += 1
        if ref_top100_rank is not None:
            hard_source_counts["hybrid_top100"] += 1
        if ref_top100_rank is None and exp_top300_rank is not None:
            hard_source_counts["expanded_only_top300"] += 1
    return hard, {
        "query_count": len(hard),
        "gold_evidence_count": gold_count,
        "hybrid_gold_rank_buckets": dict(hybrid_rank_buckets),
        "expanded_gold_rank_buckets": dict(expanded_rank_buckets),
        "source_counts": dict(hard_source_counts),
    }


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def filter_predictions(rows: list[dict[str, Any]], qids: set[str], top_k: int) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["question_id"]) in qids and int(row.get("rank", top_k + 1)) <= top_k]


def per_query_scores(qrels_by_qid: dict[str, dict[str, float]], rows_by_qid: dict[str, list[dict[str, Any]]], qid: str) -> dict[str, float]:
    gold = qrels_by_qid.get(qid, {})
    gold_ids = set(gold)
    ranked = rows_by_qid.get(qid, [])[:10]
    retrieved = [str(row["passage_id"]) for row in ranked]
    hits = gold_ids & set(retrieved)
    rr = 0.0
    for rank, pid in enumerate(retrieved, start=1):
        if pid in gold_ids:
            rr = 1.0 / rank
            break
    gains = [gold.get(pid, 0.0) for pid in retrieved]
    ideal = dcg(sorted(gold.values(), reverse=True)[:10])
    return {
        "recall@10": len(hits) / len(gold_ids) if gold_ids else 0.0,
        "mrr@10": rr,
        "ndcg@10": dcg(gains) / ideal if ideal > 0 else 0.0,
    }


def better_b_than_a(a_scores: dict[str, float], b_scores: dict[str, float], primary: str) -> bool:
    tie_order = [primary, "recall@10", "ndcg@10", "mrr@10"]
    seen = set()
    ordered = [metric for metric in tie_order if not (metric in seen or seen.add(metric))]
    for metric in ordered:
        a_value = a_scores.get(metric, 0.0)
        b_value = b_scores.get(metric, 0.0)
        if b_value > a_value:
            return True
        if b_value < a_value:
            return False
    return False


def feature(row: dict[str, Any], name: str) -> float:
    return float(row.get("metadata", {}).get("features", {}).get(name, 0.0))


FEATURE_NAMES = [
    "no_direct_mesh_overlap",
    "entity_overlap_zero",
    "max_hypergraph",
    "mean_hypergraph",
    "max_shared_cluster",
    "mean_base_rank",
    "top10_jaccard_disagreement",
    "mean_abs_rank_delta_top20",
    "candidate_pool_contains_gold_top100",
    "expanded_pool_contains_gold_top300",
    "candidate_pool_top10_misses_gold",
    "multi_evidence",
]


def rank_lookup(rows: list[dict[str, Any]], limit: int) -> dict[str, int]:
    return {str(row["passage_id"]): int(row["rank"]) for row in rows[:limit]}


def query_features(
    qid: str,
    qrels_by_qid: dict[str, dict[str, float]],
    reference_by_qid: dict[str, list[dict[str, Any]]],
    expanded_by_qid: dict[str, list[dict[str, Any]]],
    a_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
) -> dict[str, float]:
    probe_rows = b_rows[:20] or a_rows[:20]
    max_mesh = max((feature(row, "mesh_overlap_count") for row in probe_rows), default=0.0)
    max_entity = max((feature(row, "entity_overlap_count") for row in probe_rows), default=0.0)
    a_ranks = rank_lookup(a_rows, 20)
    b_ranks = rank_lookup(b_rows, 20)
    union = set(a_ranks) | set(b_ranks)
    rank_delta = [abs(float(a_ranks.get(pid, 21)) - float(b_ranks.get(pid, 21))) for pid in union]
    a_top10 = set(rank_lookup(a_rows, 10))
    b_top10 = set(rank_lookup(b_rows, 10))
    jaccard = len(a_top10 & b_top10) / len(a_top10 | b_top10) if a_top10 or b_top10 else 1.0
    gold = set(qrels_by_qid.get(qid, {}))
    ref_rows = reference_by_qid.get(qid, [])
    exp_rows = expanded_by_qid.get(qid, [])
    ref_top10 = {str(row["passage_id"]) for row in ref_rows[:10]}
    ref_top100 = {str(row["passage_id"]) for row in ref_rows[:100]}
    exp_top300 = {str(row["passage_id"]) for row in exp_rows[:300]}
    top10 = b_rows[:10] or a_rows[:10]
    return {
        "no_direct_mesh_overlap": 1.0 if max_mesh <= 0.0 else 0.0,
        "entity_overlap_zero": 1.0 if max_entity <= 0.0 else 0.0,
        "max_hypergraph": max((feature(row, "hypergraph_score_norm") for row in probe_rows), default=0.0),
        "mean_hypergraph": sum(feature(row, "hypergraph_score_norm") for row in probe_rows) / len(probe_rows) if probe_rows else 0.0,
        "max_shared_cluster": max((feature(row, "shared_mesh_term_cluster_size") for row in probe_rows), default=0.0),
        "mean_base_rank": sum(float(row.get("metadata", {}).get("base_rank") or row["rank"]) for row in top10) / len(top10) if top10 else 0.0,
        "top10_jaccard_disagreement": 1.0 - jaccard,
        "mean_abs_rank_delta_top20": sum(rank_delta) / len(rank_delta) if rank_delta else 0.0,
        "candidate_pool_contains_gold_top100": 1.0 if gold & ref_top100 else 0.0,
        "expanded_pool_contains_gold_top300": 1.0 if gold & exp_top300 else 0.0,
        "candidate_pool_top10_misses_gold": 1.0 if gold and not (gold & ref_top10) else 0.0,
        "multi_evidence": 1.0 if len(gold) > 1 else 0.0,
    }


def build_pair_examples(
    qrels_by_qid: dict[str, dict[str, float]],
    reference_by_qid: dict[str, list[dict[str, Any]]],
    expanded_by_qid: dict[str, list[dict[str, Any]]],
    a_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
    primary: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    a_by_qid = group_predictions(a_rows)
    b_by_qid = group_predictions(b_rows)
    qids = sorted(set(a_by_qid) & set(b_by_qid) & set(qrels_by_qid))
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    details: dict[str, Any] = {}
    for qid in qids:
        a_scores = per_query_scores(qrels_by_qid, a_by_qid, qid)
        b_scores = per_query_scores(qrels_by_qid, b_by_qid, qid)
        feats = query_features(qid, qrels_by_qid, reference_by_qid, expanded_by_qid, a_by_qid[qid], b_by_qid[qid])
        label = 1 if better_b_than_a(a_scores, b_scores, primary) else 0
        x_rows.append([feats[name] for name in FEATURE_NAMES])
        y_rows.append(label)
        details[qid] = {"label": label, "features": feats, "a_scores": a_scores, "b_scores": b_scores}
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64), qids, details


def make_model_candidates(seed: int) -> list[tuple[str, Any]]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    return [
        ("logistic_regression", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
        ("decision_tree_depth2", DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=seed)),
        ("decision_tree_depth3", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed)),
        ("random_forest_depth3", RandomForestClassifier(n_estimators=80, max_depth=3, class_weight="balanced", random_state=seed)),
        ("gradient_boosting", GradientBoostingClassifier(random_state=seed)),
    ]


def probability(model_name: str, model: Any, x: np.ndarray) -> np.ndarray:
    if model_name == "constant":
        return np.full((x.shape[0],), float(model), dtype=np.float64)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x).astype(np.float64)


def apply_binary_gate(
    a_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
    qids: list[str],
    labels: np.ndarray,
    *,
    top_k: int,
    retriever_name: str,
) -> list[dict[str, Any]]:
    a_by_qid = group_predictions(a_rows)
    b_by_qid = group_predictions(b_rows)
    output: list[dict[str, Any]] = []
    for qid, label in zip(qids, labels, strict=False):
        source_label = "candidate_b" if int(label) == 1 else "candidate_a"
        source = b_by_qid[qid] if int(label) == 1 else a_by_qid[qid]
        for rank, row in enumerate(source[:top_k], start=1):
            output.append(
                {
                    **row,
                    "rank": rank,
                    "retriever": retriever_name,
                    "metadata": {**row.get("metadata", {}), "gate_selected": source_label},
                }
            )
    return output


def train_and_select_gate(
    pair_name: str,
    validation_a: list[dict[str, Any]],
    validation_b: list[dict[str, Any]],
    test_a: list[dict[str, Any]],
    test_b: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    reference_by_qid: dict[str, list[dict[str, Any]]],
    expanded_by_qid: dict[str, list[dict[str, Any]]],
    primary: str,
    ks: list[int],
    seed: int,
) -> dict[str, Any]:
    qrels_by_qid = group_qrels(qrels)
    val_x, val_y, val_qids, val_details = build_pair_examples(
        qrels_by_qid, reference_by_qid, expanded_by_qid, validation_a, validation_b, primary
    )
    test_x, test_y, test_qids, test_details = build_pair_examples(
        qrels_by_qid, reference_by_qid, expanded_by_qid, test_a, test_b, primary
    )
    rng = np.random.default_rng(seed)
    idx = np.arange(len(val_qids))
    rng.shuffle(idx)
    holdout_size = max(1, int(round(len(idx) * 0.3))) if len(idx) >= 4 else len(idx)
    holdout_idx = idx[:holdout_size]
    train_idx = idx[holdout_size:] if len(idx) > holdout_size else idx

    trials: list[dict[str, Any]] = []
    if len(set(val_y[train_idx].tolist())) < 2:
        candidates = [("constant", int(val_y[train_idx][0]) if len(train_idx) else 0)]
    else:
        candidates = make_model_candidates(seed)
    holdout_qids = [val_qids[int(i)] for i in holdout_idx]
    holdout_qrels = filter_qrels(qrels, set(holdout_qids))
    for model_name, model in candidates:
        if model_name != "constant":
            model.fit(val_x[train_idx], val_y[train_idx])
        probs = probability(model_name, model, val_x[holdout_idx])
        thresholds = [0.5] if model_name == "constant" else [round(value, 2) for value in np.linspace(0.1, 0.9, 17)]
        for threshold in thresholds:
            labels = (probs >= threshold).astype(np.int64)
            preds = apply_binary_gate(validation_a, validation_b, holdout_qids, labels, top_k=max(ks), retriever_name=f"{pair_name}_holdout")
            metrics = evaluate_retrieval(holdout_qrels, preds, sorted(set(ks)))
            trial = {"model": model_name, "threshold": float(threshold), "metrics": metrics}
            if model_name == "constant":
                trial["constant_value"] = int(model)
            trials.append(trial)
    trials.sort(
        key=lambda row: (
            -float(row["metrics"].get(primary, 0.0)),
            -float(row["metrics"].get("recall@10", 0.0)),
            -float(row["metrics"].get("ndcg@10", 0.0)),
            str(row["model"]),
            float(row["threshold"]),
        )
    )
    selected = trials[0]
    if selected["model"] == "constant":
        final_model: Any = int(selected.get("constant_value", 0))
    else:
        final_model = dict(make_model_candidates(seed))[selected["model"]]
        final_model.fit(val_x, val_y)
    test_probs = probability(selected["model"], final_model, test_x)
    test_pred_labels = (test_probs >= float(selected["threshold"])).astype(np.int64)
    test_gate = apply_binary_gate(test_a, test_b, test_qids, test_pred_labels, top_k=max(ks), retriever_name=f"learned_gate_{pair_name}")
    oracle_gate = apply_binary_gate(test_a, test_b, test_qids, test_y, top_k=max(ks), retriever_name=f"oracle_gate_{pair_name}")
    test_qrels = filter_qrels(qrels, set(test_qids))
    feature_means = {}
    for label_value, label_name in [(0, "a_wins_or_ties"), (1, "b_wins")]:
        selected_idx = np.where(test_y == label_value)[0]
        if len(selected_idx):
            feature_means[label_name] = {
                name: float(np.mean(test_x[selected_idx, pos]))
                for pos, name in enumerate(FEATURE_NAMES)
            }
    return {
        "pair": pair_name,
        "feature_names": FEATURE_NAMES,
        "validation_oracle_counts": dict(Counter(int(v) for v in val_y.tolist())),
        "test_oracle_counts": dict(Counter(int(v) for v in test_y.tolist())),
        "selected": selected,
        "learned_gate_predictions": test_gate,
        "oracle_gate_predictions": oracle_gate,
        "learned_gate_metrics": evaluate_retrieval(test_qrels, test_gate, sorted(set(ks))),
        "oracle_gate_metrics": evaluate_retrieval(test_qrels, oracle_gate, sorted(set(ks))),
        "test_oracle_accuracy": float(np.mean(test_pred_labels == test_y)) if len(test_y) else 0.0,
        "test_gate_counts": dict(Counter(int(v) for v in test_pred_labels.tolist())),
        "winning_query_feature_means": feature_means,
        "test_query_details_preview": {qid: test_details[qid] for qid in test_qids[:20]},
    }


def table_row(method: str, metrics: dict[str, Any]) -> dict[str, str]:
    row = {"method": method}
    for key in ["recall@5", "recall@10", "mrr@10", "ndcg@10", "recall@100", "recall@200", "recall@300"]:
        row[key] = f"{float(metrics.get(key, 0.0)):.4f}" if key in metrics else ""
    return row


def write_table(path: str | Path, rows: list[dict[str, str]]) -> None:
    columns = ["method", "recall@5", "recall@10", "mrr@10", "ndcg@10", "recall@100", "recall@200", "recall@300"]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with target.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def hard_strata_qids(
    evaluated_hard_qids: set[str],
    qrels_by_qid: dict[str, dict[str, float]],
    reference_by_qid: dict[str, list[dict[str, Any]]],
    expanded_by_qid: dict[str, list[dict[str, Any]]],
    probe_rows: list[dict[str, Any]],
) -> dict[str, set[str]]:
    probe_by_qid = group_predictions(probe_rows)
    strata = {
        "no_direct_mesh_overlap": set(),
        "entity_overlap_zero": set(),
        "multi_evidence": set(),
        "expanded_only_top300": set(),
    }
    for qid in sorted(evaluated_hard_qids):
        rows = probe_by_qid.get(qid, [])
        feats = query_features(qid, qrels_by_qid, reference_by_qid, expanded_by_qid, rows, rows)
        for name in ["no_direct_mesh_overlap", "entity_overlap_zero", "multi_evidence"]:
            if feats.get(name, 0.0) >= 1.0:
                strata[name].add(qid)
        gold = set(qrels_by_qid.get(qid, {}))
        ref_rank = min_gold_rank(reference_by_qid.get(qid, []), gold, 100)
        exp_rank = min_gold_rank(expanded_by_qid.get(qid, []), gold, 300)
        if ref_rank is None and exp_rank is not None:
            strata["expanded_only_top300"].add(qid)
    return strata


def main() -> None:
    args = parse_args()
    qrels = read_jsonl(args.qrels)
    qrels_by_qid = group_qrels(qrels)
    reference = read_jsonl(args.reference_hybrid)
    expanded = read_jsonl(args.expanded)
    reference_by_qid = group_predictions(reference)
    expanded_by_qid = group_predictions(expanded)
    methods = {label: read_jsonl(path) for label, path in args.method}
    validation_methods = {label: read_jsonl(path) for label, path in args.validation_method}

    covered_qids = set(qrels_by_qid)
    for rows in methods.values():
        covered_qids &= set(group_predictions(rows))
    full_hard_qids, full_hard_summary = hard_subset(qrels_by_qid, reference_by_qid, expanded_by_qid, set(qrels_by_qid))
    evaluated_hard_qids, evaluated_hard_summary = hard_subset(qrels_by_qid, reference_by_qid, expanded_by_qid, covered_qids)
    hard_qrels = filter_qrels(qrels, evaluated_hard_qids)

    metrics_by_method: dict[str, Any] = {}
    table_rows: list[dict[str, str]] = []
    baseline_rows = {
        "Hybrid RRF": reference,
        "Shared-cluster expanded pool": expanded,
        **methods,
    }
    for label, rows in baseline_rows.items():
        filtered = filter_predictions(rows, evaluated_hard_qids, max(args.ks))
        metrics = evaluate_retrieval(hard_qrels, filtered, sorted(set(args.ks)))
        metrics_by_method[label] = metrics
        table_rows.append(table_row(label, metrics))

    gate_results: list[dict[str, Any]] = []
    pair_specs = [
        ("flat_vs_structural", "Flat knowledge LTR", "Retrieval + hypergraph structural LTR"),
        ("flat_vs_full", "Flat knowledge LTR", "Full KCH"),
        ("remove_hypergraph_vs_full", "Remove hypergraph", "Full KCH"),
    ]
    for pair_name, a_label, b_label in pair_specs:
        if a_label in methods and b_label in methods and a_label in validation_methods and b_label in validation_methods:
            gate_results.append(
                train_and_select_gate(
                    pair_name,
                    validation_methods[a_label],
                    validation_methods[b_label],
                    filter_predictions(methods[a_label], evaluated_hard_qids, max(args.ks)),
                    filter_predictions(methods[b_label], evaluated_hard_qids, max(args.ks)),
                    qrels,
                    reference_by_qid,
                    expanded_by_qid,
                    args.primary,
                    args.ks,
                    args.seed + len(gate_results) * 17,
                )
            )
    if gate_results:
        gate_results.sort(
            key=lambda row: (
                -float(row["learned_gate_metrics"].get(args.primary, 0.0)),
                -float(row["learned_gate_metrics"].get("recall@10", 0.0)),
                -float(row["learned_gate_metrics"].get("ndcg@10", 0.0)),
            )
        )
        selected_gate = gate_results[0]
        write_jsonl(args.output, selected_gate["learned_gate_predictions"])
        metrics_by_method["Learned gate"] = selected_gate["learned_gate_metrics"]
        metrics_by_method["Oracle gate"] = selected_gate["oracle_gate_metrics"]
        table_rows.append(table_row("Learned gate", selected_gate["learned_gate_metrics"]))
        table_rows.append(table_row("Oracle gate", selected_gate["oracle_gate_metrics"]))
    else:
        selected_gate = None

    write_table(args.table_output, table_rows)
    gate_table_rows = []
    for gate in gate_results:
        row = table_row(gate["pair"], gate["learned_gate_metrics"])
        row["method"] = f"{gate['pair']} learned"
        gate_table_rows.append(row)
        row = table_row(gate["pair"], gate["oracle_gate_metrics"])
        row["method"] = f"{gate['pair']} oracle"
        gate_table_rows.append(row)
    if gate_table_rows:
        write_table(args.gate_table_output, gate_table_rows)

    probe_label = "Full KCH" if "Full KCH" in methods else next(iter(methods))
    strata = hard_strata_qids(evaluated_hard_qids, qrels_by_qid, reference_by_qid, expanded_by_qid, methods[probe_label])
    stratum_metrics: dict[str, Any] = {}
    stratum_sources = dict(baseline_rows)
    if selected_gate:
        stratum_sources["Learned gate"] = selected_gate["learned_gate_predictions"]
        stratum_sources["Oracle gate"] = selected_gate["oracle_gate_predictions"]
    for stratum_name, stratum_qids in strata.items():
        stratum_qrels = filter_qrels(qrels, stratum_qids)
        stratum_metrics[stratum_name] = {
            "query_count": len(stratum_qids),
            "gold_evidence_count": sum(len(qrels_by_qid.get(qid, {})) for qid in stratum_qids),
            "metrics_by_method": {
                label: evaluate_retrieval(stratum_qrels, filter_predictions(rows, stratum_qids, max(args.ks)), sorted(set(args.ks)))
                for label, rows in stratum_sources.items()
            },
        }

    serializable_gate_results = []
    for gate in gate_results:
        gate_copy = {key: value for key, value in gate.items() if key not in {"learned_gate_predictions", "oracle_gate_predictions"}}
        serializable_gate_results.append(gate_copy)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "primary": args.primary,
        "reference_hybrid": args.reference_hybrid,
        "expanded": args.expanded,
        "full_hard_subset": full_hard_summary,
        "evaluated_hard_subset": evaluated_hard_summary,
        "num_method_covered_qids": len(covered_qids),
        "metrics_by_method": metrics_by_method,
        "stratum_metrics": stratum_metrics,
        "gate_results": serializable_gate_results,
        "selected_gate_pair": selected_gate["pair"] if selected_gate else None,
        "output": args.output if selected_gate else None,
    }
    write_json(args.metrics_output, payload)
    print(
        {
            "metrics": args.metrics_output,
            "table": args.table_output,
            "full_hard_qids": full_hard_summary["query_count"],
            "evaluated_hard_qids": evaluated_hard_summary["query_count"],
            "selected_gate_pair": payload["selected_gate_pair"],
        }
    )


if __name__ == "__main__":
    main()
