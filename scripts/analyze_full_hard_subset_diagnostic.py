from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.mesh_hierarchy import (
    ancestor_trees,
    descriptor_tree_numbers,
    load_mesh_hierarchy,
    parent_tree,
    tree_depth,
)
from src.rerank.hypergraph import entity_map, mesh_map
from src.utils import load_config, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose full hard subset recovered by shared-cluster top300 expansion.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--base-predictions", default="outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl")
    parser.add_argument("--expanded-predictions", default="outputs/retrieval/concept_hg_shared_clusters_full_top300.jsonl")
    parser.add_argument("--pubtator-predictions", default="outputs/retrieval/pubtator_concept_clusters_sample500_top100_preserve_top300.jsonl")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--questions", default="data/processed/bioasq_questions.jsonl")
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--mesh-hierarchy", default="data/external_knowledge/mesh_hierarchy_2026.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/full_hard_subset_diagnostic.json")
    parser.add_argument("--table-output", default="results/tables/full_hard_subset_diagnostic.md")
    parser.add_argument("--source-table-output", default="results/tables/full_hard_subset_source_breakdown.md")
    return parser.parse_args()


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    for grouped_rows in grouped.values():
        grouped_rows.sort(key=lambda item: int(item.get("rank", 10**9)))
    return dict(grouped)


def group_qrels(qrels: list[dict[str, Any]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        grouped[str(row["question_id"])].add(str(row["passage_id"]))
    return dict(grouped)


def mesh_ids(rows: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        for term in row.get("mesh_terms", []):
            mesh_ui = str(term.get("mesh_ui", "")).strip()
            if mesh_ui:
                ids.add(mesh_ui)
        mesh_ui = str(row.get("mesh_ui", "")).strip()
        if mesh_ui:
            ids.add(mesh_ui)
    return ids


def entity_ids(rows: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        for entity in row.get("entities", []):
            entity_id = str(entity.get("entity_id", "")).strip()
            if entity_id:
                ids.add(entity_id)
        entity_id = str(row.get("entity_id", "")).strip()
        if entity_id:
            ids.add(entity_id)
    return ids


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


def question_type(question: str) -> str:
    text = question.strip().lower()
    if text.startswith(("is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ", "has ", "have ")):
        return "yes_no"
    if text.startswith(("list ", "which ", "what are", "what is the list")):
        return "list"
    if text.startswith(("what ", "where ", "when ", "who ", "how many", "how much")):
        return "factoid"
    if text.startswith(("describe ", "explain ", "how ")):
        return "summary"
    return "other"


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


def min_gold_rank(rows: list[dict[str, Any]], gold: set[str], k: int) -> int | None:
    ranks = [int(row["rank"]) for row in rows[:k] if str(row["passage_id"]) in gold]
    return min(ranks) if ranks else None


def gold_rank_map(rows: list[dict[str, Any]], gold: set[str], k: int) -> dict[str, int]:
    return {str(row["passage_id"]): int(row["rank"]) for row in rows[:k] if str(row["passage_id"]) in gold}


def direct_mesh_overlap(
    qid: str,
    gold: set[str],
    q_mesh: dict[str, set[str]],
    p_mesh: dict[str, set[str]],
) -> bool:
    q_terms = q_mesh.get(qid, set())
    return any(q_terms & p_mesh.get(pid, set()) for pid in gold)


def hierarchy_mesh_overlap(
    qid: str,
    gold: set[str],
    q_hierarchy: dict[str, set[str]],
    p_hierarchy: dict[str, set[str]],
) -> bool:
    q_terms = q_hierarchy.get(qid, set())
    return any(q_terms & p_hierarchy.get(pid, set()) for pid in gold)


def entity_overlap(
    qid: str,
    gold: set[str],
    q_entities: dict[str, set[str]],
    p_entities: dict[str, set[str]],
) -> bool:
    q_terms = q_entities.get(qid, set())
    return any(q_terms & p_entities.get(pid, set()) for pid in gold)


def increment_subset(counter: Counter[str], labels: list[str]) -> None:
    for label in labels:
        counter[label] += 1


def write_markdown(path: str | Path, sections: list[tuple[str, list[tuple[str, Any]]]]) -> None:
    lines: list[str] = []
    for title, rows in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| item | value |")
        lines.append("| --- | --- |")
        for key, value in rows:
            lines.append(f"| {key} | {value} |")
        lines.append("")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def write_source_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    columns = ["source", "scope", "reason", "count"]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with target.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    qrels_path = args.qrels or paths.get("qrels", "data/processed/bioasq_qrels.jsonl")
    q_mesh_path = args.question_mesh or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")
    p_mesh_path = args.passage_mesh or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")
    q_entity_path = args.question_entities or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    p_entity_path = args.passage_entities or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")

    qrels_by_qid = group_qrels(read_jsonl(qrels_path))
    base_by_qid = group_rows(read_jsonl(args.base_predictions), "question_id")
    expanded_by_qid = group_rows(read_jsonl(args.expanded_predictions), "question_id")
    pubtator_by_qid = group_rows(read_jsonl(args.pubtator_predictions), "question_id") if Path(args.pubtator_predictions).exists() else {}
    questions = {str(row["question_id"]): str(row.get("question", "")) for row in read_jsonl(args.questions)}

    question_mesh_raw = mesh_map(read_jsonl(q_mesh_path), "question_id")
    passage_mesh_raw = mesh_map(read_jsonl(p_mesh_path), "passage_id")
    question_entity_raw = entity_map(read_jsonl(q_entity_path), "question_id")
    passage_entity_raw = entity_map(read_jsonl(p_entity_path), "passage_id")
    hierarchy = load_mesh_hierarchy(read_jsonl(args.mesh_hierarchy)) if Path(args.mesh_hierarchy).exists() else {}

    q_mesh = {qid: {f"mesh:{mesh_ui}" for mesh_ui in mesh_ids(rows)} for qid, rows in question_mesh_raw.items()}
    p_mesh = {pid: {f"mesh:{mesh_ui}" for mesh_ui in mesh_ids(rows)} for pid, rows in passage_mesh_raw.items()}
    q_hierarchy = {
        qid: {concept for mesh_ui in mesh_ids(rows) for concept in mesh_tree_concepts(mesh_ui, hierarchy)}
        for qid, rows in question_mesh_raw.items()
    }
    p_hierarchy = {
        pid: {concept for mesh_ui in mesh_ids(rows) for concept in mesh_tree_concepts(mesh_ui, hierarchy)}
        for pid, rows in passage_mesh_raw.items()
    }
    q_entities = {qid: {f"entity:{entity_id}" for entity_id in entity_ids(rows)} for qid, rows in question_entity_raw.items()}
    p_entities = {pid: {f"entity:{entity_id}" for entity_id in entity_ids(rows)} for pid, rows in passage_entity_raw.items()}

    hard_qids: set[str] = set()
    hybrid_top100_hard: set[str] = set()
    expanded_only_hard: set[str] = set()
    base_rank_distribution: Counter[str] = Counter()
    expanded_gold_rank_distribution: Counter[str] = Counter()
    expanded_only_rank_distribution: Counter[str] = Counter()
    hard_query_records: list[dict[str, Any]] = []
    shared_reason_all_added: Counter[str] = Counter()
    shared_reason_gold_added: Counter[str] = Counter()
    pubtator_reason_all_added: Counter[str] = Counter()
    pubtator_reason_gold_added: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    subset_expanded_only_counts: Counter[str] = Counter()
    possible_benefit_counts: Counter[str] = Counter()

    for qid, gold in qrels_by_qid.items():
        base_rows = base_by_qid.get(qid, [])
        expanded_rows = expanded_by_qid.get(qid, [])
        if not expanded_rows:
            continue
        base_top10_rank = min_gold_rank(base_rows, gold, 10)
        base_top100_ranks = gold_rank_map(base_rows, gold, 100)
        expanded_top300_ranks = gold_rank_map(expanded_rows, gold, 300)
        if base_top10_rank is not None:
            continue
        has_base_11_100 = any(11 <= rank <= 100 for rank in base_top100_ranks.values())
        has_expanded_101_300 = any(rank > 100 for rank in expanded_top300_ranks.values())
        if not (has_base_11_100 or has_expanded_101_300):
            continue

        hard_qids.add(qid)
        if has_base_11_100:
            hybrid_top100_hard.add(qid)
            base_rank_distribution[rank_bucket(min(base_top100_ranks.values()))] += 1
        if not has_base_11_100 and has_expanded_101_300:
            expanded_only_hard.add(qid)
            expanded_only_rank_distribution[rank_bucket(min(expanded_top300_ranks.values()))] += 1
        for rank in expanded_top300_ranks.values():
            expanded_gold_rank_distribution[rank_bucket(rank)] += 1

        no_direct_mesh = not direct_mesh_overlap(qid, gold, q_mesh, p_mesh)
        no_hierarchy_mesh = not hierarchy_mesh_overlap(qid, gold, q_hierarchy, p_hierarchy)
        entity_zero = not entity_overlap(qid, gold, q_entities, p_entities)
        multi_evidence = len(gold) >= 2
        q_type = question_type(questions.get(qid, ""))
        labels = [
            f"question_type:{q_type}",
            "no_direct_mesh_overlap" if no_direct_mesh else "has_direct_mesh_overlap",
            "no_mesh_hierarchy_match" if no_hierarchy_mesh else "has_mesh_hierarchy_match",
            "entity_overlap_zero" if entity_zero else "entity_overlap_positive",
            "multi_evidence" if multi_evidence else "single_evidence",
        ]
        increment_subset(subset_counts, labels)
        if qid in expanded_only_hard:
            increment_subset(subset_expanded_only_counts, labels)
        if no_direct_mesh or entity_zero or multi_evidence or qid in expanded_only_hard:
            increment_subset(possible_benefit_counts, labels)

        base_gold_min = min(base_top100_ranks.values()) if base_top100_ranks else None
        expanded_gold_min = min(expanded_top300_ranks.values()) if expanded_top300_ranks else None
        hard_query_records.append(
            {
                "question_id": qid,
                "question_type": q_type,
                "gold_count": len(gold),
                "base_min_gold_rank": base_gold_min,
                "expanded_min_gold_rank": expanded_gold_min,
                "category": "hybrid_top100_hard" if qid in hybrid_top100_hard else "expanded_only_hard",
                "no_direct_mesh_overlap": no_direct_mesh,
                "no_mesh_hierarchy_match": no_hierarchy_mesh,
                "entity_overlap_zero": entity_zero,
                "multi_evidence": multi_evidence,
            }
        )

        base_gold_ids = set(base_top100_ranks)
        for row in expanded_rows[:300]:
            pid = str(row["passage_id"])
            if pid in {str(base_row["passage_id"]) for base_row in base_rows[:100]}:
                continue
            reason_counts = row.get("metadata", {}).get("reason_counts", {})
            for reason, count in reason_counts.items():
                if reason != "base":
                    shared_reason_all_added[reason] += int(count)
                    if pid in gold and pid not in base_gold_ids:
                        shared_reason_gold_added[reason] += int(count)
        for row in pubtator_by_qid.get(qid, [])[:300]:
            pid = str(row["passage_id"])
            if pid in {str(base_row["passage_id"]) for base_row in base_rows[:100]}:
                continue
            reason_counts = row.get("metadata", {}).get("reason_counts", {})
            for reason, count in reason_counts.items():
                if reason != "base":
                    pubtator_reason_all_added[reason] += int(count)
                    if pid in gold and pid not in base_gold_ids:
                        pubtator_reason_gold_added[reason] += int(count)

    hard_gold_count = sum(len(qrels_by_qid[qid]) for qid in hard_qids)
    source_rows: list[dict[str, Any]] = []
    for source, scope, counter in [
        ("shared_cluster", "all_added_candidates_in_hard_top300", shared_reason_all_added),
        ("shared_cluster", "new_gold_added_in_hard_top300", shared_reason_gold_added),
        ("pubtator_sample500", "all_added_candidates_in_overlap_hard_top300", pubtator_reason_all_added),
        ("pubtator_sample500", "new_gold_added_in_overlap_hard_top300", pubtator_reason_gold_added),
    ]:
        for reason, count in counter.most_common():
            source_rows.append({"source": source, "scope": scope, "reason": reason, "count": count})

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "base_predictions": args.base_predictions,
        "expanded_predictions": args.expanded_predictions,
        "pubtator_predictions": args.pubtator_predictions if pubtator_by_qid else None,
        "hard_subset": {
            "query_count": len(hard_qids),
            "gold_count": hard_gold_count,
            "hybrid_top100_hard_queries": len(hybrid_top100_hard),
            "expanded_only_hard_queries": len(expanded_only_hard),
        },
        "base_rank_distribution": dict(base_rank_distribution),
        "expanded_gold_rank_distribution": dict(expanded_gold_rank_distribution),
        "expanded_only_gold_rank_distribution": dict(expanded_only_rank_distribution),
        "source_reason_counts": {
            "shared_cluster_all_added": dict(shared_reason_all_added),
            "shared_cluster_new_gold_added": dict(shared_reason_gold_added),
            "pubtator_sample500_all_added": dict(pubtator_reason_all_added),
            "pubtator_sample500_new_gold_added": dict(pubtator_reason_gold_added),
        },
        "stratified_counts": dict(subset_counts),
        "expanded_only_stratified_counts": dict(subset_expanded_only_counts),
        "expanded_only_stratified_rates": {
            key: subset_expanded_only_counts[key] / value for key, value in subset_counts.items() if value
        },
        "likely_expansion_benefit_counts": dict(possible_benefit_counts),
        "hard_query_preview": sorted(hard_query_records, key=lambda row: (row["category"], row["question_id"]))[:50],
    }
    write_json(args.metrics_output, payload)

    sections = [
        (
            "Full hard subset",
            [
                ("hard subset query count", len(hard_qids)),
                ("hard subset gold count", hard_gold_count),
                ("Hybrid top100 hard queries", len(hybrid_top100_hard)),
                ("expanded-only hard queries", len(expanded_only_hard)),
            ],
        ),
        ("Base rank buckets", sorted(base_rank_distribution.items())),
        ("Expanded gold rank buckets", sorted(expanded_gold_rank_distribution.items())),
        ("Expanded-only rank buckets", sorted(expanded_only_rank_distribution.items())),
        ("Stratified hard query counts", sorted(subset_counts.items())),
        ("Expanded-only hard query counts", sorted(subset_expanded_only_counts.items())),
        (
            "Expanded-only share within each stratum",
            sorted(
                (key, f"{subset_expanded_only_counts[key] / value:.3f}")
                for key, value in subset_counts.items()
                if value
            ),
        ),
        ("Likely expansion benefit signals", sorted(possible_benefit_counts.items())),
    ]
    write_markdown(args.table_output, sections)
    write_source_table(args.source_table_output, source_rows)
    print(
        {
            "metrics": args.metrics_output,
            "table": args.table_output,
            "source_table": args.source_table_output,
            "hard_queries": len(hard_qids),
            "hybrid_top100_hard": len(hybrid_top100_hard),
            "expanded_only": len(expanded_only_hard),
        }
    )


if __name__ == "__main__":
    main()
