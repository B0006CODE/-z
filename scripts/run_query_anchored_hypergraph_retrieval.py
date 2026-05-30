from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval, group_predictions, group_qrels
from src.knowledge.mesh_hierarchy import (
    ancestor_trees,
    descriptor_tree_numbers,
    load_mesh_hierarchy,
    parent_tree,
    tree_depth,
)
from src.rerank.hypergraph import entity_map, mesh_map, relations_map
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pure query-anchored hypergraph candidate generation for BioASQ evidence retrieval."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--bm25-predictions", default="outputs/retrieval/bm25_full_top100.jsonl")
    parser.add_argument("--dense-predictions", default="outputs/retrieval/dense_full_top100.jsonl")
    parser.add_argument("--hybrid-predictions", default="outputs/retrieval/hybrid_full_top100.jsonl")
    parser.add_argument("--enhanced-predictions", default="outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl")
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--mesh-hierarchy", default="data/external_knowledge/mesh_hierarchy_2026.jsonl")
    parser.add_argument("--relations", default=None)
    parser.add_argument("--output", default="outputs/retrieval/query_anchored_hg_test_top100.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/query_anchored_hg_test_metrics.json")
    parser.add_argument("--table-output", default="results/tables/query_anchored_hg_test.md")
    parser.add_argument("--subset-table-output", default="results/tables/query_anchored_hg_test_subsets.md")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-qids", type=int, default=None)
    parser.add_argument("--qid-selection-order", choices=["lexical", "numeric"], default="lexical")
    parser.add_argument("--target-split", choices=["all", "validation", "test"], default="test")
    parser.add_argument("--split-modulo", type=int, default=5)
    parser.add_argument("--validation-remainders", type=int, nargs="+", default=[3])
    parser.add_argument("--test-remainders", type=int, nargs="+", default=[4])
    parser.add_argument("--max-df-ratio", type=float, default=0.10)
    parser.add_argument("--max-passages-per-hyperedge", type=int, default=400)
    parser.add_argument("--min-tree-depth", type=int, default=3)
    parser.add_argument("--mesh-weight", type=float, default=1.0)
    parser.add_argument("--entity-weight", type=float, default=0.85)
    parser.add_argument("--hierarchy-weight", type=float, default=0.35)
    parser.add_argument("--relation-weight", type=float, default=0.55)
    parser.add_argument("--major-topic-multiplier", type=float, default=1.25)
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


def selected_qids(all_qids: set[str], args: argparse.Namespace) -> set[str]:
    qids = set(all_qids)
    if args.target_split == "validation":
        remainders = set(args.validation_remainders)
        qids = {qid for qid in qids if qid_bucket(qid, args.split_modulo) in remainders}
    elif args.target_split == "test":
        remainders = set(args.test_remainders)
        qids = {qid for qid in qids if qid_bucket(qid, args.split_modulo) in remainders}
    if args.max_qids is not None:
        qids = set(sorted(qids, key=lambda qid: qid_sort_key(qid, args.qid_selection_order))[: args.max_qids])
    return qids


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def filter_predictions(rows: list[dict[str, Any]], qids: set[str], top_k: int) -> list[dict[str, Any]]:
    grouped = group_predictions(rows)
    output: list[dict[str, Any]] = []
    for qid in sorted(qids):
        for rank, row in enumerate(grouped.get(qid, [])[:top_k], start=1):
            output.append({**row, "rank": rank})
    return output


def mesh_ids(rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    seen: set[str] = set()
    for row in rows:
        mesh_ui = str(row.get("mesh_ui", "")).strip()
        if not mesh_ui or mesh_ui in seen:
            continue
        seen.add(mesh_ui)
        multiplier = 1.0
        if bool(row.get("major_topic", False)):
            multiplier = 1.25
        values.append((mesh_ui, multiplier))
    return values


def entity_ids(rows: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        entity_id = str(row.get("entity_id", "")).strip()
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        ids.append(entity_id)
    return ids


def mesh_tree_concepts(mesh_ui: str, hierarchy: dict[str, Any], min_depth: int) -> set[str]:
    concepts: set[str] = set()
    for tree_number in descriptor_tree_numbers(mesh_ui, hierarchy):
        if tree_depth(tree_number) >= min_depth:
            concepts.add(f"mesh_tree:{tree_number}")
        parent = parent_tree(tree_number)
        if parent and tree_depth(parent) >= min_depth:
            concepts.add(f"mesh_parent:{parent}")
        for ancestor in ancestor_trees(tree_number, include_self=False):
            if tree_depth(ancestor) >= min_depth:
                concepts.add(f"mesh_ancestor:{ancestor}")
    return concepts


def passage_concepts(
    passage_id: str,
    passage_entities: dict[str, list[dict[str, Any]]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    hierarchy: dict[str, Any],
    min_tree_depth: int,
) -> set[str]:
    concepts: set[str] = set()
    for entity_id in entity_ids(passage_entities.get(passage_id, [])):
        concepts.add(f"entity:{entity_id}")
    for mesh_ui, _multiplier in mesh_ids(passage_mesh.get(passage_id, [])):
        concepts.add(f"mesh:{mesh_ui}")
        concepts.update(mesh_tree_concepts(mesh_ui, hierarchy, min_tree_depth))
    return concepts


def build_concept_index(
    passage_entities: dict[str, list[dict[str, Any]]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    hierarchy: dict[str, Any],
    min_tree_depth: int,
) -> tuple[dict[str, list[str]], Counter[str], int]:
    concept_to_passages: dict[str, set[str]] = defaultdict(set)
    all_pids = sorted(set(passage_entities) | set(passage_mesh))
    for passage_id in all_pids:
        for concept in passage_concepts(passage_id, passage_entities, passage_mesh, hierarchy, min_tree_depth):
            concept_to_passages[concept].add(passage_id)
    indexed = {concept: sorted(pids) for concept, pids in concept_to_passages.items()}
    df = Counter({concept: len(pids) for concept, pids in indexed.items()})
    return indexed, df, len(all_pids)


def query_anchors(
    qid: str,
    question_entities: dict[str, list[dict[str, Any]]],
    question_mesh: dict[str, list[dict[str, Any]]],
    relations: dict[str, list[dict[str, Any]]],
    hierarchy: dict[str, Any],
    args: argparse.Namespace,
) -> list[tuple[str, str, float]]:
    anchors: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()

    def add(reason: str, concept: str, weight: float) -> None:
        key = (reason, concept)
        if concept and key not in seen:
            seen.add(key)
            anchors.append((reason, concept, weight))

    q_entities = entity_ids(question_entities.get(qid, []))
    for entity_id in q_entities:
        add("query_entity", f"entity:{entity_id}", args.entity_weight)
    for mesh_ui, multiplier in mesh_ids(question_mesh.get(qid, [])):
        add("query_mesh", f"mesh:{mesh_ui}", args.mesh_weight * multiplier * args.major_topic_multiplier)
        for concept in sorted(mesh_tree_concepts(mesh_ui, hierarchy, args.min_tree_depth)):
            add("query_mesh_hierarchy", concept, args.hierarchy_weight)
    for source_entity_id in q_entities:
        for relation in relations.get(source_entity_id, []):
            target_entity_id = str(relation.get("target_entity_id", "")).strip()
            if target_entity_id:
                add("query_primekg_relation", f"entity:{target_entity_id}", args.relation_weight)
    return anchors


def generate_hypergraph_predictions(
    qids: set[str],
    question_entities: dict[str, list[dict[str, Any]]],
    question_mesh: dict[str, list[dict[str, Any]]],
    passage_entities: dict[str, list[dict[str, Any]]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    hierarchy: dict[str, Any],
    relations: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    concept_to_passages, df, passage_count = build_concept_index(
        passage_entities,
        passage_mesh,
        hierarchy,
        args.min_tree_depth,
    )
    max_df = max(1, int(passage_count * args.max_df_ratio))
    predictions: list[dict[str, Any]] = []
    stats = Counter()
    per_query_preview: list[dict[str, Any]] = []

    for qid in sorted(qids, key=lambda value: qid_sort_key(value, args.qid_selection_order)):
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, Counter[str]] = defaultdict(Counter)
        active_anchors = query_anchors(qid, question_entities, question_mesh, relations, hierarchy, args)
        stats["queries_seen"] += 1
        stats["anchors_total"] += len(active_anchors)
        for reason, concept, weight in active_anchors:
            concept_df = df.get(concept, 0)
            if concept_df <= 0 or concept_df > max_df:
                stats[f"skipped_{reason}"] += 1
                continue
            stats[f"activated_{reason}"] += 1
            specificity = math.log1p(passage_count / max(concept_df, 1))
            contribution = weight * specificity
            for passage_id in concept_to_passages.get(concept, [])[: args.max_passages_per_hyperedge]:
                scores[passage_id] += contribution
                reasons[passage_id][reason] += 1
        ranked = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))[: args.top_k]
        if ranked:
            stats["queries_with_predictions"] += 1
        if len(per_query_preview) < 20:
            per_query_preview.append(
                {
                    "question_id": qid,
                    "num_anchors": len(active_anchors),
                    "num_candidates": len(scores),
                    "num_ranked": len(ranked),
                    "top_reason_counts": dict(reasons[ranked[0][0]]) if ranked else {},
                }
            )
        for rank, (passage_id, score) in enumerate(ranked, start=1):
            predictions.append(
                {
                    "question_id": qid,
                    "passage_id": passage_id,
                    "rank": rank,
                    "score": float(score),
                    "retriever": "query_anchored_hypergraph_retrieval",
                    "metadata": {
                        "reason_counts": dict(reasons[passage_id]),
                        "max_df": max_df,
                        "max_df_ratio": args.max_df_ratio,
                    },
                }
            )
    diagnostics = {
        "num_indexed_passages": passage_count,
        "num_indexed_concepts": len(concept_to_passages),
        "max_df": max_df,
        "stats": dict(stats),
        "per_query_preview": per_query_preview,
    }
    return predictions, diagnostics


def hard_subset_qids(qrels: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], qids: set[str]) -> set[str]:
    qrels_by_qid = group_qrels(qrels)
    baseline_by_qid = group_predictions(baseline_rows)
    hard: set[str] = set()
    for qid in qids:
        gold = set(qrels_by_qid.get(qid, {}))
        ranked = [str(row["passage_id"]) for row in baseline_by_qid.get(qid, [])[:100]]
        if gold and not gold.intersection(ranked[:10]) and gold.intersection(ranked):
            hard.add(qid)
    return hard


def direct_mesh_overlap_subset(
    qrels: list[dict[str, Any]],
    question_mesh: dict[str, list[dict[str, Any]]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    qids: set[str],
) -> set[str]:
    qrels_by_qid = group_qrels(qrels)
    no_overlap: set[str] = set()
    for qid in qids:
        q_mesh = {mesh_ui for mesh_ui, _multiplier in mesh_ids(question_mesh.get(qid, []))}
        gold_ids = set(qrels_by_qid.get(qid, {}))
        has_overlap = False
        for pid in gold_ids:
            p_mesh = {mesh_ui for mesh_ui, _multiplier in mesh_ids(passage_mesh.get(pid, []))}
            if q_mesh & p_mesh:
                has_overlap = True
                break
        if not has_overlap:
            no_overlap.add(qid)
    return no_overlap


def new_gold_diagnostics(
    qrels: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    qids: set[str],
    *,
    k: int,
) -> dict[str, Any]:
    qrels_by_qid = group_qrels(qrels)
    candidate_by_qid = group_predictions(candidate_rows)
    reference_by_qid = group_predictions(reference_rows)
    new_gold: set[tuple[str, str]] = set()
    query_ids: set[str] = set()
    reason_counts: Counter[str] = Counter()
    for qid in qids:
        gold = set(qrels_by_qid.get(qid, {}))
        if not gold:
            continue
        reference_ids = {str(row["passage_id"]) for row in reference_by_qid.get(qid, [])[:k]}
        for row in candidate_by_qid.get(qid, [])[:k]:
            pid = str(row["passage_id"])
            if pid in gold and pid not in reference_ids:
                new_gold.add((qid, pid))
                query_ids.add(qid)
                for reason, count in row.get("metadata", {}).get("reason_counts", {}).items():
                    reason_counts[reason] += int(count)
    return {
        f"new_gold_evidence_not_in_reference@{k}": len(new_gold),
        f"queries_with_new_gold_not_in_reference@{k}": len(query_ids),
        f"new_gold_reason_counts@{k}": dict(reason_counts),
    }


def add_evidence_coverage(metrics: dict[str, Any], ks: list[int]) -> dict[str, Any]:
    output = dict(metrics)
    for k in ks:
        if f"recall@{k}" in output:
            output[f"evidence_coverage@{k}"] = output[f"recall@{k}"]
    return output


def metric_row(subset: str, method: str, metrics: dict[str, Any]) -> dict[str, str]:
    row = {"subset": subset, "method": method}
    for key in ["recall@5", "recall@10", "mrr@10", "ndcg@10", "evidence_coverage@10", "recall@100", "mrr@100", "ndcg@100"]:
        row[key] = f"{float(metrics.get(key, 0.0)):.4f}" if key in metrics else ""
    return row


def write_table(path: str | Path, rows: list[dict[str, str]], include_subset: bool) -> None:
    columns = ["method", "recall@5", "recall@10", "mrr@10", "ndcg@10", "evidence_coverage@10", "recall@100", "mrr@100", "ndcg@100"]
    if include_subset:
        columns = ["subset", *columns]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with target.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
    all_qids = set(group_qrels(qrels))
    qids = selected_qids(all_qids, args)
    eval_qrels = filter_qrels(qrels, qids)

    bm25_rows = filter_predictions(read_jsonl(args.bm25_predictions), qids, args.top_k)
    dense_rows = filter_predictions(read_jsonl(args.dense_predictions), qids, args.top_k)
    hybrid_rows = filter_predictions(read_jsonl(args.hybrid_predictions), qids, args.top_k)
    enhanced_rows = filter_predictions(read_jsonl(args.enhanced_predictions), qids, args.top_k)
    question_entities = entity_map(read_jsonl(question_entities_path), "question_id")
    passage_entities = entity_map(read_jsonl(passage_entities_path), "passage_id")
    question_mesh = mesh_map(read_jsonl(question_mesh_path), "question_id")
    passage_mesh = mesh_map(read_jsonl(passage_mesh_path), "passage_id")
    hierarchy = load_mesh_hierarchy(read_jsonl(args.mesh_hierarchy)) if Path(args.mesh_hierarchy).exists() else {}
    relations = relations_map(read_jsonl(relations_path)) if Path(relations_path).exists() else {}

    hypergraph_rows, hypergraph_diagnostics = generate_hypergraph_predictions(
        qids,
        question_entities,
        question_mesh,
        passage_entities,
        passage_mesh,
        hierarchy,
        relations,
        args,
    )
    write_jsonl(args.output, hypergraph_rows)

    ks = sorted(set(args.ks))
    method_rows = {
        "BM25": bm25_rows,
        "Dense": dense_rows,
        "Hybrid RRF": hybrid_rows,
        "Enhanced Hybrid w122": enhanced_rows,
        "Query-Anchored Hypergraph Retrieval": hypergraph_rows,
    }
    overall_metrics = {
        name: add_evidence_coverage(evaluate_retrieval(eval_qrels, rows, ks), ks)
        for name, rows in method_rows.items()
    }
    hard_qids = hard_subset_qids(qrels, enhanced_rows, qids)
    no_direct_mesh_qids = direct_mesh_overlap_subset(qrels, question_mesh, passage_mesh, qids)
    subset_qids = {
        "hard_subset": hard_qids,
        "no_direct_mesh_overlap": no_direct_mesh_qids,
    }
    subset_metrics: dict[str, dict[str, Any]] = {}
    subset_rows: list[dict[str, str]] = []
    for subset_name, current_qids in subset_qids.items():
        subset_qrels = filter_qrels(qrels, current_qids)
        subset_metrics[subset_name] = {}
        for method_name, rows in method_rows.items():
            metrics = add_evidence_coverage(evaluate_retrieval(subset_qrels, filter_predictions(rows, current_qids, args.top_k), ks), ks)
            subset_metrics[subset_name][method_name] = metrics
            subset_rows.append(metric_row(subset_name, method_name, metrics))

    diagnostics = {
        "vs_hybrid": new_gold_diagnostics(qrels, hypergraph_rows, hybrid_rows, qids, k=args.top_k),
        "vs_enhanced_hybrid": new_gold_diagnostics(qrels, hypergraph_rows, enhanced_rows, qids, k=args.top_k),
    }
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "target_split": args.target_split,
        "split": {
            "modulo": args.split_modulo,
            "validation_remainders": args.validation_remainders,
            "test_remainders": args.test_remainders,
            "max_qids": args.max_qids,
            "qid_selection_order": args.qid_selection_order,
            "num_eval_qids": len(qids),
            "num_hard_subset_qids": len(hard_qids),
            "num_no_direct_mesh_overlap_qids": len(no_direct_mesh_qids),
        },
        "paths": {
            "qrels": qrels_path,
            "bm25_predictions": args.bm25_predictions,
            "dense_predictions": args.dense_predictions,
            "hybrid_predictions": args.hybrid_predictions,
            "enhanced_predictions": args.enhanced_predictions,
            "output": args.output,
        },
        "hyperparameters": {
            "top_k": args.top_k,
            "max_df_ratio": args.max_df_ratio,
            "max_passages_per_hyperedge": args.max_passages_per_hyperedge,
            "min_tree_depth": args.min_tree_depth,
            "mesh_weight": args.mesh_weight,
            "entity_weight": args.entity_weight,
            "hierarchy_weight": args.hierarchy_weight,
            "relation_weight": args.relation_weight,
        },
        "hypergraph_diagnostics": hypergraph_diagnostics,
        "overall_metrics": overall_metrics,
        "subset_metrics": subset_metrics,
        "new_gold_diagnostics": diagnostics,
    }
    write_json(args.metrics_output, payload)
    write_table(args.table_output, [metric_row("overall", name, metrics) for name, metrics in overall_metrics.items()], include_subset=False)
    write_table(args.subset_table_output, subset_rows, include_subset=True)
    write_json(Path(paths.get("logs_dir", "logs")) / "run_query_anchored_hypergraph_retrieval_summary.json", payload)
    print(
        {
            "output": args.output,
            "metrics": args.metrics_output,
            "table": args.table_output,
            "num_eval_qids": len(qids),
            "hypergraph_recall@100": overall_metrics["Query-Anchored Hypergraph Retrieval"].get("recall@100"),
            "new_gold_vs_hybrid@100": diagnostics["vs_hybrid"].get("new_gold_evidence_not_in_reference@100"),
            "new_gold_vs_enhanced@100": diagnostics["vs_enhanced_hybrid"].get("new_gold_evidence_not_in_reference@100"),
        }
    )


if __name__ == "__main__":
    main()
