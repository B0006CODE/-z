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
from src.rerank.hypergraph import entity_map, mesh_map
from src.utils import load_config, read_jsonl, set_seed, write_json


VALID_PUBTATOR_TYPES = {"disease", "chemical", "gene", "species", "mutation", "cellline"}
MEDICAL_STOP_CONCEPT_IDS = {"", "-", "the", "with", "and", "or", "of", "in", "to", "for", "a", "an"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PubTator3 filtered candidate expansion ablations without writing large prediction JSONL files."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--base-predictions", default="outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--question-pubtator", default="data/processed/bioasq_question_pubtator_concepts.jsonl")
    parser.add_argument("--passage-pubtator", default="data/processed/bioasq_passage_pubtator_concepts.jsonl")
    parser.add_argument("--mesh-hierarchy", default="data/external_knowledge/mesh_hierarchy_2026.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/pubtator_filtered_expansion_metrics.json")
    parser.add_argument("--table-output", default="results/tables/pubtator_filtered_expansion.md")
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--base-keep", type=int, default=100)
    parser.add_argument("--max-qids", type=int, default=10)
    parser.add_argument("--qid-selection-order", choices=["lexical", "numeric"], default="numeric")
    parser.add_argument("--max-expansion-per-concept", type=int, default=250)
    parser.add_argument("--max-shared-concepts", type=int, default=80)
    parser.add_argument("--direct-weight", type=float, default=0.36)
    parser.add_argument("--cluster-weight", type=float, default=0.12)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--ks", type=int, nargs="+", default=[100, 200, 300])
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


def pubtator_map(rows: list[dict[str, Any]], id_key: str) -> dict[str, list[dict[str, Any]]]:
    return {str(row[id_key]): list(row.get("concepts", [])) for row in rows}


def pubtator_key(concept: dict[str, Any], allowed_types: set[str] | None) -> str | None:
    concept_type = str(concept.get("type", "")).strip().lower().replace(" ", "")
    concept_id = str(concept.get("concept_id", "")).strip()
    if allowed_types is not None and concept_type not in allowed_types:
        return None
    if concept_type not in VALID_PUBTATOR_TYPES:
        return None
    if concept_id.lower() in MEDICAL_STOP_CONCEPT_IDS:
        return None
    return f"pubtator:{concept_type}:{concept_id}"


def pubtator_ids(rows: list[dict[str, Any]], allowed_types: set[str] | None) -> set[str]:
    concepts: set[str] = set()
    for row in rows:
        key = pubtator_key(row, allowed_types)
        if key:
            concepts.add(key)
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


def build_passage_indexes(
    passage_pubtator: dict[str, list[dict[str, Any]]],
    passage_mesh: dict[str, list[dict[str, Any]]],
    passage_entities: dict[str, list[dict[str, Any]]],
    hierarchy: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, set[str]], dict[str, set[str]], dict[str, set[str]], dict[str, set[str]], Counter[str]]:
    concept_to_passages: dict[str, set[str]] = defaultdict(set)
    pubtator_by_pid: dict[str, set[str]] = {}
    mesh_by_pid: dict[str, set[str]] = {}
    mesh_hierarchy_by_pid: dict[str, set[str]] = {}
    entity_by_pid: dict[str, set[str]] = {}
    all_pids = sorted(set(passage_pubtator) | set(passage_mesh) | set(passage_entities))
    for pid in all_pids:
        pubtator = pubtator_ids(passage_pubtator.get(pid, []), allowed_types=None)
        meshes = mesh_ids(passage_mesh.get(pid, []))
        entities = entity_ids(passage_entities.get(pid, []))
        hierarchy_concepts = {concept for mesh_ui in meshes for concept in mesh_tree_concepts(mesh_ui, hierarchy)}
        pubtator_by_pid[pid] = pubtator
        mesh_by_pid[pid] = {f"mesh:{mesh_ui}" for mesh_ui in meshes}
        mesh_hierarchy_by_pid[pid] = hierarchy_concepts
        entity_by_pid[pid] = {f"entity:{entity_id}" for entity_id in entities}
        for concept in pubtator:
            concept_to_passages[concept].add(pid)
    df = Counter({concept: len(pids) for concept, pids in concept_to_passages.items()})
    return (
        {concept: sorted(pids) for concept, pids in concept_to_passages.items()},
        pubtator_by_pid,
        mesh_by_pid,
        mesh_hierarchy_by_pid,
        entity_by_pid,
        df,
    )


def build_question_features(
    qids: set[str],
    question_pubtator: dict[str, list[dict[str, Any]]],
    question_mesh: dict[str, list[dict[str, Any]]],
    question_entities: dict[str, list[dict[str, Any]]],
    hierarchy: dict[str, Any],
    allowed_types: set[str] | None,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    q_pubtator: dict[str, set[str]] = {}
    q_mesh: dict[str, set[str]] = {}
    q_mesh_hierarchy: dict[str, set[str]] = {}
    q_entity: dict[str, set[str]] = {}
    for qid in qids:
        pubtator = pubtator_ids(question_pubtator.get(qid, []), allowed_types=allowed_types)
        meshes = mesh_ids(question_mesh.get(qid, []))
        entities = entity_ids(question_entities.get(qid, []))
        q_pubtator[qid] = pubtator
        q_mesh[qid] = {f"mesh:{mesh_ui}" for mesh_ui in meshes}
        q_mesh_hierarchy[qid] = {concept for mesh_ui in meshes for concept in mesh_tree_concepts(mesh_ui, hierarchy)}
        q_entity[qid] = {f"entity:{entity_id}" for entity_id in entities}
    return q_pubtator, q_mesh, q_mesh_hierarchy, q_entity


def add_matches(
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


def rare_concepts(concepts: set[str], df: Counter[str], limit: int, allowed_types: set[str] | None) -> set[str]:
    filtered = {concept for concept in concepts if concept_type(concept) in allowed_types} if allowed_types else set(concepts)
    return {
        concept
        for concept, _count in sorted(
            ((concept, df.get(concept, 10**9)) for concept in filtered),
            key=lambda item: (item[1], item[0]),
        )[:limit]
    }


def concept_type(concept: str) -> str:
    parts = concept.split(":")
    return parts[1] if len(parts) > 2 and parts[0] == "pubtator" else ""


def setting_grid() -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = [
        {
            "name": "pubtator_direct_only",
            "use_direct": True,
            "use_shared": False,
            "allowed_types": None,
            "max_df_ratio": 0.20,
            "seed_top_n": 25,
            "combined_filter": False,
        },
        {
            "name": "pubtator_shared_only",
            "use_direct": False,
            "use_shared": True,
            "allowed_types": None,
            "max_df_ratio": 0.20,
            "seed_top_n": 25,
            "combined_filter": False,
        },
        {
            "name": "pubtator_disease_gene_only",
            "use_direct": True,
            "use_shared": True,
            "allowed_types": {"disease", "gene"},
            "max_df_ratio": 0.20,
            "seed_top_n": 25,
            "combined_filter": False,
        },
        {
            "name": "pubtator_disease_chemical_gene_only",
            "use_direct": True,
            "use_shared": True,
            "allowed_types": {"disease", "chemical", "gene"},
            "max_df_ratio": 0.20,
            "seed_top_n": 25,
            "combined_filter": False,
        },
        {
            "name": "pubtator_combined_medical_filter",
            "use_direct": True,
            "use_shared": True,
            "allowed_types": {"disease", "chemical", "gene"},
            "max_df_ratio": 0.05,
            "seed_top_n": 10,
            "combined_filter": True,
        },
    ]
    for ratio in [0.01, 0.02, 0.05, 0.10]:
        settings.append(
            {
                "name": f"pubtator_low_df_{ratio:.2f}",
                "use_direct": True,
                "use_shared": True,
                "allowed_types": None,
                "max_df_ratio": ratio,
                "seed_top_n": 25,
                "combined_filter": False,
            }
        )
    for seed_top_n in [5, 10, 20, 25]:
        settings.append(
            {
                "name": f"pubtator_seed_top_{seed_top_n}",
                "use_direct": True,
                "use_shared": True,
                "allowed_types": None,
                "max_df_ratio": 0.05,
                "seed_top_n": seed_top_n,
                "combined_filter": False,
            }
        )
    for ratio in [0.01, 0.02, 0.05, 0.10]:
        for seed_top_n in [5, 10, 20, 25]:
            settings.append(
                {
                    "name": f"pubtator_dcg_lowdf_{ratio:.2f}_seed{seed_top_n}",
                    "use_direct": True,
                    "use_shared": True,
                    "allowed_types": {"disease", "chemical", "gene"},
                    "max_df_ratio": ratio,
                    "seed_top_n": seed_top_n,
                    "combined_filter": False,
                }
            )
            settings.append(
                {
                    "name": f"pubtator_combined_lowdf_{ratio:.2f}_seed{seed_top_n}",
                    "use_direct": True,
                    "use_shared": True,
                    "allowed_types": {"disease", "chemical", "gene"},
                    "max_df_ratio": ratio,
                    "seed_top_n": seed_top_n,
                    "combined_filter": True,
                }
            )
    dedup: dict[str, dict[str, Any]] = {}
    for setting in settings:
        dedup[setting["name"]] = setting
    return list(dedup.values())


def filter_qrels(qrels: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in qids]


def medical_filter_reasons(
    qid: str,
    pid: str,
    shared_seed_concepts: set[str],
    base_rank_lookup: dict[str, int],
    q_mesh: dict[str, set[str]],
    q_mesh_hierarchy: dict[str, set[str]],
    q_entity: dict[str, set[str]],
    pubtator_by_pid: dict[str, set[str]],
    mesh_by_pid: dict[str, set[str]],
    mesh_hierarchy_by_pid: dict[str, set[str]],
    entity_by_pid: dict[str, set[str]],
) -> list[str]:
    reasons: list[str] = []
    if q_mesh.get(qid, set()) & mesh_by_pid.get(pid, set()):
        reasons.append("mesh_overlap")
    if q_mesh_hierarchy.get(qid, set()) & mesh_hierarchy_by_pid.get(pid, set()):
        reasons.append("mesh_hierarchy_match")
    if q_entity.get(qid, set()) & entity_by_pid.get(pid, set()):
        reasons.append("entity_overlap")
    if shared_seed_concepts & pubtator_by_pid.get(pid, set()):
        reasons.append("shared_candidate_pubtator_cluster")
    if base_rank_lookup.get(pid, 10**9) <= 200:
        reasons.append("base_rank_le_200")
    return reasons


def build_expanded_predictions(
    setting: dict[str, Any],
    *,
    base_by_qid: dict[str, list[dict[str, Any]]],
    concept_to_passages: dict[str, list[str]],
    pubtator_by_pid: dict[str, set[str]],
    mesh_by_pid: dict[str, set[str]],
    mesh_hierarchy_by_pid: dict[str, set[str]],
    entity_by_pid: dict[str, set[str]],
    q_pubtator: dict[str, set[str]],
    q_mesh: dict[str, set[str]],
    q_mesh_hierarchy: dict[str, set[str]],
    q_entity: dict[str, set[str]],
    df: Counter[str],
    num_pubtator_passages: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    max_df = max(1, int(num_pubtator_passages * float(setting["max_df_ratio"])))
    allowed_types = setting["allowed_types"]
    expanded: list[dict[str, Any]] = []
    reason_totals: Counter[str] = Counter()
    for qid in sorted(base_by_qid):
        base_rows = sorted(base_by_qid[qid], key=lambda row: int(row["rank"]))[: args.base_keep]
        base_pid_set = {str(row["passage_id"]) for row in base_rows}
        base_rank_lookup = {str(row["passage_id"]): int(row["rank"]) for row in base_by_qid[qid]}
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, Counter[str]] = defaultdict(Counter)
        for row in base_rows:
            pid = str(row["passage_id"])
            rank = int(row["rank"])
            scores[pid] += 1.0 / (args.rrf_k + rank)
            reasons[pid]["base"] += 1

        if setting["use_direct"]:
            direct_concepts = {
                concept for concept in q_pubtator.get(qid, set()) if allowed_types is None or concept_type(concept) in allowed_types
            }
            add_matches(
                scores,
                reasons,
                concept_to_passages,
                df,
                direct_concepts,
                reason="pubtator_direct_concept",
                weight=args.direct_weight,
                max_df=max_df,
                max_per_concept=args.max_expansion_per_concept,
            )

        shared_seed_concepts: set[str] = set()
        if setting["use_shared"]:
            for seed_row in base_rows[: int(setting["seed_top_n"])]:
                seed_pid = str(seed_row["passage_id"])
                seed_rank = int(seed_row["rank"])
                seed_concepts = rare_concepts(
                    pubtator_by_pid.get(seed_pid, set()),
                    df,
                    args.max_shared_concepts,
                    allowed_types,
                )
                shared_seed_concepts |= seed_concepts
                local_weight = args.cluster_weight / (args.rrf_k + seed_rank)
                add_matches(
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

        expansion_ranked = []
        medical_filter_reason_counts: Counter[str] = Counter()
        for pid, score in sorted(scores.items(), key=lambda item: (-item[1], str(item[0]))):
            if pid in base_pid_set:
                continue
            if setting["combined_filter"]:
                filter_reasons = medical_filter_reasons(
                    qid,
                    pid,
                    shared_seed_concepts,
                    base_rank_lookup,
                    q_mesh,
                    q_mesh_hierarchy,
                    q_entity,
                    pubtator_by_pid,
                    mesh_by_pid,
                    mesh_hierarchy_by_pid,
                    entity_by_pid,
                )
                if not filter_reasons:
                    continue
                for reason in filter_reasons:
                    medical_filter_reason_counts[reason] += 1
                    reasons[pid][f"filter_{reason}"] += 1
            expansion_ranked.append((pid, score))

        ranked = [(str(row["passage_id"]), float(scores[str(row["passage_id"])])) for row in base_rows]
        ranked.extend(expansion_ranked)
        ranked = ranked[: args.top_k]
        for rank, (pid, score) in enumerate(ranked, start=1):
            reason_counts = dict(reasons[pid])
            for reason, count in reason_counts.items():
                reason_totals[reason] += int(count)
            expanded.append(
                {
                    "question_id": qid,
                    "passage_id": pid,
                    "rank": rank,
                    "score": float(score),
                    "metadata": {
                        "base_rank": base_rank_lookup.get(pid),
                        "reason_counts": reason_counts,
                    },
                }
            )
        reason_totals.update({f"combined_filter_{key}": value for key, value in medical_filter_reason_counts.items()})
    return expanded, reason_totals


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


def table_row(setting: dict[str, Any], metrics: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    expanded_metrics = metrics["expanded_metrics"]
    return {
        "setting": setting["name"],
        "types": ",".join(sorted(setting["allowed_types"])) if setting["allowed_types"] else "all",
        "direct": str(setting["use_direct"]),
        "shared": str(setting["use_shared"]),
        "combined_filter": str(setting["combined_filter"]),
        "max_df_ratio": setting["max_df_ratio"],
        "seed_top_n": setting["seed_top_n"],
        "recall@100": f"{expanded_metrics.get('recall@100', 0.0):.4f}",
        "recall@200": f"{expanded_metrics.get('recall@200', 0.0):.4f}",
        "recall@300": f"{expanded_metrics.get('recall@300', 0.0):.4f}",
        "new_gold@200": quality.get("new_gold_evidence_recovered@200", 0),
        "new_gold@300": quality.get("new_gold_evidence_recovered@300", 0),
        "queries_new_gold@200": quality.get("queries_with_new_gold_recovered@200", 0),
        "queries_new_gold@300": quality.get("queries_with_new_gold_recovered@300", 0),
        "added@200": quality.get("added_candidates@200", 0),
        "added@300": quality.get("added_candidates@300", 0),
        "added_non_gold@200": quality.get("added_non_gold_candidates@200", 0),
        "added_non_gold@300": quality.get("added_non_gold_candidates@300", 0),
        "noise@200": f"{quality.get('noise_ratio@200', 0.0):.6f}",
        "noise@300": f"{quality.get('noise_ratio@300', 0.0):.6f}",
    }


def write_tables(path: str | Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "setting",
        "types",
        "direct",
        "shared",
        "combined_filter",
        "max_df_ratio",
        "seed_top_n",
        "recall@100",
        "recall@200",
        "recall@300",
        "new_gold@200",
        "new_gold@300",
        "queries_new_gold@200",
        "queries_new_gold@300",
        "added@200",
        "added@300",
        "added_non_gold@200",
        "added_non_gold@300",
        "noise@200",
        "noise@300",
    ]
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
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(seed)
    paths = config["paths"]

    qrels_path = args.qrels or paths.get("qrels", "data/processed/bioasq_qrels.jsonl")
    q_mesh_path = args.question_mesh or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")
    p_mesh_path = args.passage_mesh or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")
    q_entity_path = args.question_entities or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    p_entity_path = args.passage_entities or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")

    base_predictions = read_jsonl(args.base_predictions)
    selected_qids = set(
        sorted(
            {str(row["question_id"]) for row in base_predictions},
            key=lambda qid: qid_sort_key(qid, args.qid_selection_order),
        )[: args.max_qids]
    )
    base_predictions = [row for row in base_predictions if str(row["question_id"]) in selected_qids]
    base_by_qid = group_rows(base_predictions, "question_id")
    qrels = filter_qrels(read_jsonl(qrels_path), selected_qids)

    question_mesh = mesh_map(read_jsonl(q_mesh_path), "question_id")
    passage_mesh = mesh_map(read_jsonl(p_mesh_path), "passage_id")
    question_entities = entity_map(read_jsonl(q_entity_path), "question_id")
    passage_entities = entity_map(read_jsonl(p_entity_path), "passage_id")
    question_pubtator = pubtator_map(read_jsonl(args.question_pubtator), "question_id")
    passage_pubtator = pubtator_map(read_jsonl(args.passage_pubtator), "passage_id")
    hierarchy = load_mesh_hierarchy(read_jsonl(args.mesh_hierarchy)) if Path(args.mesh_hierarchy).exists() else {}

    concept_to_passages, pubtator_by_pid, mesh_by_pid, mesh_hierarchy_by_pid, entity_by_pid, df = build_passage_indexes(
        passage_pubtator,
        passage_mesh,
        passage_entities,
        hierarchy,
    )
    qids = set(base_by_qid)
    q_pubtator_all, q_mesh, q_mesh_hierarchy, q_entity = build_question_features(
        qids,
        question_pubtator,
        question_mesh,
        question_entities,
        hierarchy,
        allowed_types=None,
    )

    base_metrics = evaluate_retrieval(qrels, base_predictions, args.ks)
    settings = setting_grid()
    setting_payloads: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for setting in settings:
        q_pubtator, _, _, _ = build_question_features(
            qids,
            question_pubtator,
            question_mesh,
            question_entities,
            hierarchy,
            allowed_types=setting["allowed_types"],
        )
        expanded, reason_totals = build_expanded_predictions(
            setting,
            base_by_qid=base_by_qid,
            concept_to_passages=concept_to_passages,
            pubtator_by_pid=pubtator_by_pid,
            mesh_by_pid=mesh_by_pid,
            mesh_hierarchy_by_pid=mesh_hierarchy_by_pid,
            entity_by_pid=entity_by_pid,
            q_pubtator=q_pubtator,
            q_mesh=q_mesh,
            q_mesh_hierarchy=q_mesh_hierarchy,
            q_entity=q_entity,
            df=df,
            num_pubtator_passages=len(pubtator_by_pid),
            args=args,
        )
        expanded_metrics = evaluate_retrieval(qrels, expanded, args.ks)
        quality = expansion_quality(qrels, base_predictions, expanded, base_keep=args.base_keep, ks=args.ks)
        payload = {
            "setting": {
                **setting,
                "allowed_types": sorted(setting["allowed_types"]) if setting["allowed_types"] else None,
            },
            "expansion_reason_counts": dict(reason_totals),
            "base_metrics": base_metrics,
            "expanded_metrics": expanded_metrics,
            "expansion_quality": quality,
        }
        setting_payloads.append(payload)
        rows.append(table_row(setting, payload, quality))

    # Primary sort: keep recovered evidence, then lower noise, then higher recall@300.
    rows.sort(
        key=lambda row: (
            -int(row["new_gold@300"]),
            float(row["noise@300"]),
            -float(row["recall@300"]),
            row["setting"],
        )
    )
    best_rows = rows[:10]
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "base_predictions": args.base_predictions,
        "max_qids": args.max_qids,
        "qid_selection_order": args.qid_selection_order,
        "num_questions": len(qids),
        "num_qrels": len(qrels),
        "num_pubtator_passages": len(pubtator_by_pid),
        "num_pubtator_concepts": len(concept_to_passages),
        "base_keep": args.base_keep,
        "top_k": args.top_k,
        "preserve_base_ranks": True,
        "base_metrics": base_metrics,
        "best_rows": best_rows,
        "settings": setting_payloads,
        "question_pubtator_concept_counts": {
            "questions_with_any": sum(1 for concepts in q_pubtator_all.values() if concepts),
            "total_concepts": sum(len(concepts) for concepts in q_pubtator_all.values()),
        },
    }
    write_json(args.metrics_output, payload)
    write_tables(args.table_output, rows)
    print(
        {
            "metrics": args.metrics_output,
            "table": args.table_output,
            "num_questions": len(qids),
            "best": best_rows[:3],
        }
    )


if __name__ == "__main__":
    main()
