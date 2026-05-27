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

from src.evaluation.retrieval_metrics import evaluate_retrieval
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
        description="Build a concept-normalized MeSH/entity hypergraph expansion candidate pool."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--base-predictions", default="outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--question-cui", default=None, help="Optional UMLS CUI annotations from build_umls_concepts.py.")
    parser.add_argument("--passage-cui", default=None, help="Optional UMLS CUI annotations from build_umls_concepts.py.")
    parser.add_argument("--question-pubtator", default=None, help="Optional PubTator3 annotations from build_pubtator3_concepts.py.")
    parser.add_argument("--passage-pubtator", default=None, help="Optional PubTator3 annotations from build_pubtator3_concepts.py.")
    parser.add_argument("--mesh-hierarchy", default="data/external_knowledge/mesh_hierarchy_2026.jsonl")
    parser.add_argument("--relations", default=None)
    parser.add_argument("--output", default="outputs/retrieval/concept_hypergraph_expanded_top200.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/concept_hypergraph_expanded_metrics.json")
    parser.add_argument("--table-output", default="results/tables/concept_hypergraph_expansion.md")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--base-keep", type=int, default=100)
    parser.add_argument(
        "--preserve-base-ranks",
        action="store_true",
        help="Keep original top base-keep candidates first and append concept-expanded candidates after them.",
    )
    parser.add_argument("--seed-top-n", type=int, default=25)
    parser.add_argument("--max-qids", type=int, default=None)
    parser.add_argument("--qid-selection-order", choices=["lexical", "numeric"], default="lexical")
    parser.add_argument("--max-expansion-per-concept", type=int, default=250)
    parser.add_argument("--max-shared-concepts", type=int, default=80)
    parser.add_argument("--max-concept-df-ratio", type=float, default=0.20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--direct-mesh-weight", type=float, default=0.42)
    parser.add_argument("--direct-entity-weight", type=float, default=0.26)
    parser.add_argument("--direct-cui-weight", type=float, default=0.36)
    parser.add_argument("--mesh-hierarchy-weight", type=float, default=0.16)
    parser.add_argument("--cluster-weight", type=float, default=0.12)
    parser.add_argument("--relation-weight", type=float, default=0.18)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            "query_mesh_exact",
            "mesh_hierarchy",
            "shared_candidate_concept_clusters",
            "entity_overlap_clusters",
            "primekg_relation",
            "cui_exact",
            "pubtator_concept_clusters",
        ],
        help=(
            "Expansion sources to enable. Options: query_mesh_exact, mesh_hierarchy, "
            "shared_candidate_concept_clusters, entity_overlap_clusters, primekg_relation, cui_exact, "
            "pubtator_concept_clusters."
        ),
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100, 200, 300])
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(grouped)


def qid_sort_key(qid: str, order: str) -> tuple[int, int | str]:
    if order == "numeric" and qid.isdigit():
        return (0, int(qid))
    return (1, qid)


def mesh_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("mesh_ui", "")).strip() for row in rows if str(row.get("mesh_ui", "")).strip()}


def entity_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("entity_id", "")).strip() for row in rows if str(row.get("entity_id", "")).strip()}


def cui_map(rows: list[dict[str, Any]], id_key: str) -> dict[str, list[dict[str, Any]]]:
    return {str(row[id_key]): list(row.get("cui_concepts", [])) for row in rows}


def cui_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("cui", "")).strip() for row in rows if str(row.get("cui", "")).strip()}


def pubtator_map(rows: list[dict[str, Any]], id_key: str) -> dict[str, list[dict[str, Any]]]:
    return {str(row[id_key]): list(row.get("concepts", [])) for row in rows}


def pubtator_ids(rows: list[dict[str, Any]]) -> set[str]:
    concepts = set()
    for row in rows:
        concept_type = str(row.get("type", "")).strip().lower().replace(" ", "_")
        concept_id = str(row.get("concept_id", "")).strip()
        if concept_type and concept_id:
            concepts.add(f"pubtator:{concept_type}:{concept_id}")
    return concepts


def mesh_tree_concepts(mesh_ui: str, hierarchy: dict[str, Any], *, min_depth: int = 3) -> set[str]:
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
    passage_mesh: dict[str, list[dict[str, Any]]],
    passage_entities: dict[str, list[dict[str, Any]]],
    passage_cui: dict[str, list[dict[str, Any]]],
    passage_pubtator: dict[str, list[dict[str, Any]]],
    hierarchy: dict[str, Any],
) -> dict[str, set[str]]:
    meshes = mesh_ids(passage_mesh.get(passage_id, []))
    entities = entity_ids(passage_entities.get(passage_id, []))
    cuis = cui_ids(passage_cui.get(passage_id, []))
    pubtator = pubtator_ids(passage_pubtator.get(passage_id, []))
    hierarchy_concepts = {concept for mesh_ui in meshes for concept in mesh_tree_concepts(mesh_ui, hierarchy)}
    return {
        "mesh": {f"mesh:{mesh_ui}" for mesh_ui in meshes},
        "entity": {f"entity:{entity_id}" for entity_id in entities},
        "cui": {f"cui:{cui}" for cui in cuis},
        "pubtator": pubtator,
        "hierarchy": hierarchy_concepts,
        "all": {f"mesh:{mesh_ui}" for mesh_ui in meshes}
        | {f"entity:{entity_id}" for entity_id in entities}
        | {f"cui:{cui}" for cui in cuis}
        | pubtator
        | hierarchy_concepts,
    }


def question_concepts(
    qid: str,
    question_mesh: dict[str, list[dict[str, Any]]],
    question_entities: dict[str, list[dict[str, Any]]],
    question_cui: dict[str, list[dict[str, Any]]],
    question_pubtator: dict[str, list[dict[str, Any]]],
    hierarchy: dict[str, Any],
) -> dict[str, set[str]]:
    meshes = mesh_ids(question_mesh.get(qid, []))
    entities = entity_ids(question_entities.get(qid, []))
    cuis = cui_ids(question_cui.get(qid, []))
    pubtator = pubtator_ids(question_pubtator.get(qid, []))
    hierarchy_concepts = {concept for mesh_ui in meshes for concept in mesh_tree_concepts(mesh_ui, hierarchy)}
    return {
        "mesh": {f"mesh:{mesh_ui}" for mesh_ui in meshes},
        "entity": {f"entity:{entity_id}" for entity_id in entities},
        "cui": {f"cui:{cui}" for cui in cuis},
        "pubtator": pubtator,
        "hierarchy": hierarchy_concepts,
        "all": {f"mesh:{mesh_ui}" for mesh_ui in meshes}
        | {f"entity:{entity_id}" for entity_id in entities}
        | {f"cui:{cui}" for cui in cuis}
        | pubtator
        | hierarchy_concepts,
    }


def build_indexes(
    passage_mesh: dict[str, list[dict[str, Any]]],
    passage_entities: dict[str, list[dict[str, Any]]],
    passage_cui: dict[str, list[dict[str, Any]]],
    passage_pubtator: dict[str, list[dict[str, Any]]],
    hierarchy: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, dict[str, set[str]]], Counter[str]]:
    concept_to_passages: dict[str, set[str]] = defaultdict(set)
    concept_by_passage: dict[str, dict[str, set[str]]] = {}
    all_pids = sorted(set(passage_mesh) | set(passage_entities) | set(passage_cui) | set(passage_pubtator))
    for pid in all_pids:
        concepts = passage_concepts(pid, passage_mesh, passage_entities, passage_cui, passage_pubtator, hierarchy)
        concept_by_passage[pid] = concepts
        for concept in concepts["all"]:
            concept_to_passages[concept].add(pid)
    df = Counter({concept: len(pids) for concept, pids in concept_to_passages.items()})
    return {concept: sorted(pids) for concept, pids in concept_to_passages.items()}, concept_by_passage, df


def add_concept_matches(
    scores: dict[str, float],
    reasons: dict[str, Counter[str]],
    concept_to_passages: dict[str, list[str]],
    df: Counter[str],
    concepts: set[str],
    *,
    reason: str,
    weight: float,
    max_df: int,
    max_per_concept: int,
) -> None:
    for concept in sorted(concepts):
        concept_df = df.get(concept, 0)
        if concept_df <= 0 or concept_df > max_df:
            continue
        bonus = weight / math.sqrt(concept_df)
        for pid in concept_to_passages.get(concept, [])[:max_per_concept]:
            scores[pid] += bonus
            reasons[pid][reason] += 1


def relation_target_entities(q_entities: set[str], relations: dict[str, list[dict[str, Any]]]) -> set[str]:
    targets: set[str] = set()
    for entity_id in q_entities:
        for relation in relations.get(entity_id, []):
            target = str(relation.get("target_entity_id", "")).strip()
            if target:
                targets.add(f"entity:{target}")
    return targets


def rare_concepts(concepts: set[str], df: Counter[str], limit: int) -> set[str]:
    return {
        concept
        for concept, _count in sorted(
            ((concept, df.get(concept, 10**9)) for concept in concepts),
            key=lambda item: (item[1], item[0]),
        )[:limit]
    }


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def metric_row(name: str, metrics: dict[str, Any]) -> dict[str, str]:
    row = {"method": name}
    for key in ["recall@10", "recall@20", "recall@50", "recall@100", "recall@200", "recall@300", "mrr@10", "ndcg@10"]:
        row[key] = f"{float(metrics.get(key, 0.0)):.4f}" if key in metrics else ""
    return row


def write_markdown_table(path: str | Path, rows: list[dict[str, str]]) -> None:
    columns = ["method", "recall@10", "recall@20", "recall@50", "recall@100", "recall@200", "recall@300", "mrr@10", "ndcg@10"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = target.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def expansion_quality(
    qrels: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
    expanded_predictions: list[dict[str, Any]],
    *,
    base_keep: int,
    ks: list[int],
) -> dict[str, Any]:
    gold_by_qid: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        gold_by_qid[str(row["question_id"])].add(str(row["passage_id"]))
    base_by_qid = group_rows(base_predictions, "question_id")
    expanded_by_qid = group_rows(expanded_predictions, "question_id")
    diagnostics: dict[str, Any] = {}
    for k in sorted(set(ks)):
        new_gold: set[tuple[str, str]] = set()
        added = 0
        added_non_gold = 0
        queries_with_new_gold: set[str] = set()
        new_gold_reason_counts: Counter[str] = Counter()
        for qid, expanded_rows in expanded_by_qid.items():
            gold = gold_by_qid.get(qid, set())
            base_ids = {
                str(row["passage_id"])
                for row in sorted(base_by_qid.get(qid, []), key=lambda item: int(item["rank"]))[:base_keep]
            }
            for row in sorted(expanded_rows, key=lambda item: int(item["rank"]))[:k]:
                pid = str(row["passage_id"])
                if pid in base_ids:
                    continue
                added += 1
                if pid in gold:
                    new_gold.add((qid, pid))
                    queries_with_new_gold.add(qid)
                    for reason, count in row.get("metadata", {}).get("reason_counts", {}).items():
                        if reason != "base":
                            new_gold_reason_counts[reason] += int(count)
                else:
                    added_non_gold += 1
        diagnostics[f"new_gold_evidence_recovered@{k}"] = len(new_gold)
        diagnostics[f"queries_with_new_gold_recovered@{k}"] = len(queries_with_new_gold)
        diagnostics[f"added_candidates@{k}"] = added
        diagnostics[f"added_non_gold_candidates@{k}"] = added_non_gold
        diagnostics[f"noise_ratio@{k}"] = added_non_gold / added if added else 0.0
        diagnostics[f"new_gold_reason_counts@{k}"] = dict(new_gold_reason_counts)
    return diagnostics


def main() -> None:
    args = parse_args()
    enabled_sources = set(args.sources)
    supported_sources = {
        "query_mesh_exact",
        "mesh_hierarchy",
        "shared_candidate_concept_clusters",
        "entity_overlap_clusters",
        "primekg_relation",
        "cui_exact",
        "pubtator_concept_clusters",
    }
    unsupported = enabled_sources - supported_sources
    if unsupported:
        raise ValueError(f"Unsupported expansion sources: {sorted(unsupported)}. Supported: {sorted(supported_sources)}")
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(seed)
    paths = config["paths"]

    qrels_path = args.qrels or paths.get("qrels", "data/processed/bioasq_qrels.jsonl")
    q_mesh_path = args.question_mesh or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")
    p_mesh_path = args.passage_mesh or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")
    q_entity_path = args.question_entities or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    p_entity_path = args.passage_entities or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")
    relations_path = args.relations or paths.get("primekg_relations", "data/external_knowledge/primekg_project_relations.jsonl")

    base_predictions = read_jsonl(args.base_predictions)
    if args.max_qids is not None:
        selected = set(
            sorted(
                {str(row["question_id"]) for row in base_predictions},
                key=lambda qid: qid_sort_key(qid, args.qid_selection_order),
            )[: args.max_qids]
        )
        base_predictions = [row for row in base_predictions if str(row["question_id"]) in selected]

    question_mesh = mesh_map(read_jsonl(q_mesh_path), "question_id")
    passage_mesh = mesh_map(read_jsonl(p_mesh_path), "passage_id")
    question_entities = entity_map(read_jsonl(q_entity_path), "question_id")
    passage_entities = entity_map(read_jsonl(p_entity_path), "passage_id")
    question_cui = cui_map(read_jsonl(args.question_cui), "question_id") if args.question_cui and Path(args.question_cui).exists() else {}
    passage_cui = cui_map(read_jsonl(args.passage_cui), "passage_id") if args.passage_cui and Path(args.passage_cui).exists() else {}
    question_pubtator = pubtator_map(read_jsonl(args.question_pubtator), "question_id") if args.question_pubtator and Path(args.question_pubtator).exists() else {}
    passage_pubtator = pubtator_map(read_jsonl(args.passage_pubtator), "passage_id") if args.passage_pubtator and Path(args.passage_pubtator).exists() else {}
    hierarchy = load_mesh_hierarchy(read_jsonl(args.mesh_hierarchy)) if Path(args.mesh_hierarchy).exists() else {}
    relations = relations_map(read_jsonl(relations_path)) if Path(relations_path).exists() else {}
    qrels = read_jsonl(qrels_path)

    concept_to_passages, concept_by_passage, df = build_indexes(passage_mesh, passage_entities, passage_cui, passage_pubtator, hierarchy)
    max_df = max(1, int(len(concept_by_passage) * args.max_concept_df_ratio))
    base_by_qid = group_rows(base_predictions, "question_id")
    expanded: list[dict[str, Any]] = []
    expansion_stats = Counter()
    per_query_stats: list[dict[str, Any]] = []

    for qid in sorted(base_by_qid):
        q_concepts = question_concepts(qid, question_mesh, question_entities, question_cui, question_pubtator, hierarchy)
        q_entity_set = {concept.removeprefix("entity:") for concept in q_concepts["entity"]}
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, Counter[str]] = defaultdict(Counter)
        base_rows = sorted(base_by_qid[qid], key=lambda row: int(row["rank"]))[: args.base_keep]

        for row in base_rows:
            pid = str(row["passage_id"])
            rank = int(row["rank"])
            scores[pid] += args.base_weight / (args.rrf_k + rank)
            reasons[pid]["base"] += 1

        if "query_mesh_exact" in enabled_sources:
            add_concept_matches(
                scores,
                reasons,
                concept_to_passages,
                df,
                q_concepts["mesh"],
                reason="direct_mesh",
                weight=args.direct_mesh_weight,
                max_df=max_df,
                max_per_concept=args.max_expansion_per_concept,
            )
        if "entity_overlap_clusters" in enabled_sources:
            add_concept_matches(
                scores,
                reasons,
                concept_to_passages,
                df,
                q_concepts["entity"],
                reason="direct_entity",
                weight=args.direct_entity_weight,
                max_df=max_df,
                max_per_concept=args.max_expansion_per_concept,
            )
        if "cui_exact" in enabled_sources:
            add_concept_matches(
                scores,
                reasons,
                concept_to_passages,
                df,
                q_concepts["cui"],
                reason="direct_cui",
                weight=args.direct_cui_weight,
                max_df=max_df,
                max_per_concept=args.max_expansion_per_concept,
            )
        if "pubtator_concept_clusters" in enabled_sources:
            add_concept_matches(
                scores,
                reasons,
                concept_to_passages,
                df,
                q_concepts["pubtator"],
                reason="pubtator_direct_concept",
                weight=args.direct_cui_weight,
                max_df=max_df,
                max_per_concept=args.max_expansion_per_concept,
            )
        if "mesh_hierarchy" in enabled_sources:
            add_concept_matches(
                scores,
                reasons,
                concept_to_passages,
                df,
                q_concepts["hierarchy"],
                reason="mesh_hierarchy",
                weight=args.mesh_hierarchy_weight,
                max_df=max_df,
                max_per_concept=args.max_expansion_per_concept,
            )
        if "primekg_relation" in enabled_sources:
            add_concept_matches(
                scores,
                reasons,
                concept_to_passages,
                df,
                relation_target_entities(q_entity_set, relations),
                reason="primekg_relation",
                weight=args.relation_weight,
                max_df=max_df,
                max_per_concept=args.max_expansion_per_concept,
            )

        if "shared_candidate_concept_clusters" in enabled_sources:
            for seed_row in base_rows[: args.seed_top_n]:
                seed_pid = str(seed_row["passage_id"])
                seed_rank = int(seed_row["rank"])
                seed_concepts = rare_concepts(
                    concept_by_passage.get(seed_pid, {}).get("all", set()),
                    df,
                    args.max_shared_concepts,
                )
                local_weight = args.cluster_weight / (args.rrf_k + seed_rank)
                add_concept_matches(
                    scores,
                    reasons,
                    concept_to_passages,
                    df,
                    seed_concepts,
                    reason="shared_candidate_concept",
                    weight=local_weight,
                    max_df=max_df,
                    max_per_concept=args.max_expansion_per_concept,
                )
        if "pubtator_concept_clusters" in enabled_sources:
            for seed_row in base_rows[: args.seed_top_n]:
                seed_pid = str(seed_row["passage_id"])
                seed_rank = int(seed_row["rank"])
                seed_concepts = rare_concepts(
                    concept_by_passage.get(seed_pid, {}).get("pubtator", set()),
                    df,
                    args.max_shared_concepts,
                )
                local_weight = args.cluster_weight / (args.rrf_k + seed_rank)
                add_concept_matches(
                    scores,
                    reasons,
                    concept_to_passages,
                    df,
                    seed_concepts,
                    reason="pubtator_concept_cluster",
                    weight=local_weight,
                    max_df=max_df,
                    max_per_concept=args.max_expansion_per_concept,
                )

        base_pid_set = {str(row["passage_id"]) for row in base_rows}
        if args.preserve_base_ranks:
            base_ranked = [(str(row["passage_id"]), float(scores[str(row["passage_id"])])) for row in base_rows]
            expansion_ranked = [
                (pid, score)
                for pid, score in sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
                if pid not in base_pid_set
            ]
            ranked = (base_ranked + expansion_ranked)[: args.top_k]
        else:
            ranked = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))[: args.top_k]
        added = 0
        for rank, (pid, score) in enumerate(ranked, start=1):
            reason_counts = dict(reasons[pid])
            if pid not in base_pid_set:
                added += 1
            for reason, count in reason_counts.items():
                expansion_stats[reason] += count
            expanded.append(
                {
                    "question_id": qid,
                    "passage_id": pid,
                    "rank": rank,
                    "score": float(score),
                    "retriever": "concept_hypergraph_expansion",
                    "metadata": {
                        "base_rank": next((int(row["rank"]) for row in base_rows if str(row["passage_id"]) == pid), None),
                        "reason_counts": reason_counts,
                        "source_scores": {"concept_hypergraph_expansion": float(score)},
                        "source_ranks": {"concept_hypergraph_expansion": rank},
                        "source_retrievers": {"concept_hypergraph_expansion": "concept_hypergraph_expansion"},
                    },
                }
            )
        per_query_stats.append(
            {
                "question_id": qid,
                "num_ranked": len(ranked),
                "num_base_kept": len(base_rows),
                "num_added": added,
                "num_question_mesh_concepts": len(q_concepts["mesh"]),
                "num_question_entity_concepts": len(q_concepts["entity"]),
                "num_question_cui_concepts": len(q_concepts["cui"]),
                "num_question_hierarchy_concepts": len(q_concepts["hierarchy"]),
            }
        )

    write_jsonl(args.output, expanded)
    qids = {str(row["question_id"]) for row in expanded}
    eval_qrels = filter_qrels(qrels, qids)
    base_metrics = evaluate_retrieval(eval_qrels, base_predictions, args.ks)
    expanded_metrics = evaluate_retrieval(eval_qrels, expanded, args.ks)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "base_predictions": args.base_predictions,
        "output": args.output,
        "top_k": args.top_k,
        "base_keep": args.base_keep,
        "preserve_base_ranks": args.preserve_base_ranks,
        "sources": sorted(enabled_sources),
        "seed_top_n": args.seed_top_n,
        "max_df": max_df,
        "num_questions": len(qids),
        "num_predictions": len(expanded),
        "num_passages_with_concepts": len(concept_by_passage),
        "num_concepts": len(concept_to_passages),
        "question_cui": args.question_cui,
        "passage_cui": args.passage_cui,
        "question_pubtator": args.question_pubtator,
        "passage_pubtator": args.passage_pubtator,
        "mean_added_per_query": sum(row["num_added"] for row in per_query_stats) / len(per_query_stats) if per_query_stats else 0.0,
        "expansion_reason_counts": dict(expansion_stats),
        "expansion_quality": expansion_quality(
            eval_qrels,
            base_predictions,
            expanded,
            base_keep=args.base_keep,
            ks=args.ks,
        ),
        "base_metrics": base_metrics,
        "expanded_metrics": expanded_metrics,
        "per_query_stats_preview": per_query_stats[:20],
    }
    write_json(args.metrics_output, payload)
    write_markdown_table(args.table_output, [metric_row("Enhanced Hybrid w122", base_metrics), metric_row("Concept-Hypergraph Expansion", expanded_metrics)])
    write_json(Path(paths.get("logs_dir", "logs")) / "run_concept_hypergraph_expansion_summary.json", payload)
    print(
        {
            "output": args.output,
            "metrics": args.metrics_output,
            "num_questions": len(qids),
            "base_recall@100": base_metrics.get("recall@100"),
            "expanded_recall@100": expanded_metrics.get("recall@100"),
            "expanded_recall@200": expanded_metrics.get("recall@200"),
        }
    )


if __name__ == "__main__":
    main()
