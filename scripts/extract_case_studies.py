from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import group_predictions, group_qrels
from src.utils import read_jsonl, write_json


FEATURE_NAMES = [
    "base_rank_score",
    "hybrid_score",
    "hypergraph_score_norm",
    "entity_overlap_count",
    "question_entity_coverage",
    "mesh_overlap_count",
    "question_mesh_coverage",
    "primekg_relation_count",
    "question_relation_coverage",
    "local_num_hyperedges",
    "local_shared_entity_edges",
    "local_document_mesh_edges",
    "local_primekg_relation_edges",
]


def parse_dataset_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Dataset input must use Dataset=prefix format.")
    label, prefix = value.split("=", 1)
    label = label.strip()
    prefix = prefix.strip()
    if not label or not prefix:
        raise argparse.ArgumentTypeError("Dataset label and prefix must be non-empty.")
    return label, prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract concise HGB reranking case studies.")
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_arg,
        required=True,
        help=(
            "Dataset=prefix where prefix is bioasq or pubmedqa. "
            "Default paths are inferred from the project layout."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-cases-per-dataset", type=int, default=3)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def default_paths(prefix: str) -> dict[str, str]:
    if prefix == "bioasq":
        return {
            "questions": "data/processed/bioasq_questions.jsonl",
            "corpus": "data/processed/bioasq_corpus.jsonl",
            "qrels": "data/processed/bioasq_qrels.jsonl",
            "baseline": "outputs/retrieval/hybrid_full_top100.jsonl",
            "candidate": "outputs/rerank/learning_hgb_all_test_top100.jsonl",
            "question_entities": "data/processed/bioasq_question_entities.jsonl",
            "passage_entities": "data/processed/bioasq_passage_entities.jsonl",
            "question_mesh": "data/processed/bioasq_question_mesh.jsonl",
            "passage_mesh": "data/processed/bioasq_passage_mesh.jsonl",
        }
    if prefix == "pubmedqa":
        return {
            "questions": "data/processed/pubmedqa_pqa_labeled_questions.jsonl",
            "corpus": "data/processed/pubmedqa_pqa_labeled_corpus.jsonl",
            "qrels": "data/processed/pubmedqa_pqa_labeled_qrels.jsonl",
            "baseline": "outputs/retrieval/pubmedqa_hybrid_full_top100.jsonl",
            "candidate": "outputs/rerank/pubmedqa_learning_hgb_test_top100.jsonl",
            "question_entities": "data/processed/pubmedqa_question_entities.jsonl",
            "passage_entities": "data/processed/pubmedqa_passage_entities.jsonl",
            "question_mesh": "data/processed/pubmedqa_pqa_labeled_question_mesh.jsonl",
            "passage_mesh": "data/processed/pubmedqa_pqa_labeled_passage_mesh.jsonl",
        }
    raise ValueError(f"Unsupported dataset prefix: {prefix}")


def index_by_id(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows}


def rank_map(rows_by_qid: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        qid: {str(row["passage_id"]): row for row in rows}
        for qid, rows in rows_by_qid.items()
    }


def entity_map(rows: list[dict[str, Any]], id_key: str) -> dict[str, set[str]]:
    mapped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        values = {
            str(entity.get("canonical", "")).strip()
            for entity in row.get("entities", [])
            if str(entity.get("canonical", "")).strip()
        }
        mapped[str(row[id_key])] = values
    return dict(mapped)


def mesh_map(rows: list[dict[str, Any]], id_key: str) -> dict[str, set[str]]:
    mapped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        values = {
            str(term.get("normalized") or term.get("mesh_name") or "").strip()
            for term in row.get("mesh_terms", [])
            if str(term.get("normalized") or term.get("mesh_name") or "").strip()
        }
        mapped[str(row[id_key])] = values
    return dict(mapped)


def min_gold_rank(rank_by_pid: dict[str, dict[str, Any]], gold_ids: set[str]) -> tuple[int | None, str | None]:
    ranked_gold = [
        (int(row["rank"]), pid)
        for pid, row in rank_by_pid.items()
        if pid in gold_ids
    ]
    if not ranked_gold:
        return None, None
    return min(ranked_gold, key=lambda item: item[0])


def truncate(text: str, max_chars: int = 360) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def feature_values(row: dict[str, Any] | None) -> dict[str, float]:
    if not row:
        return {}
    features = row.get("metadata", {}).get("features", {})
    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def build_case(
    label: str,
    qid: str,
    selected_pid: str,
    baseline_rank_value: int | None,
    hgb_rank_value: int,
    rescued: bool,
    questions: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    candidate_rank: dict[str, dict[str, dict[str, Any]]],
    baseline_rank: dict[str, dict[str, dict[str, Any]]],
    q_entities: dict[str, set[str]],
    p_entities: dict[str, set[str]],
    q_mesh: dict[str, set[str]],
    p_mesh: dict[str, set[str]],
) -> dict[str, Any]:
    hgb_row = candidate_rank.get(qid, {}).get(selected_pid)
    base_row = baseline_rank.get(qid, {}).get(selected_pid)
    q_ent = q_entities.get(qid, set())
    p_ent = p_entities.get(selected_pid, set())
    q_mesh_set = q_mesh.get(qid, set())
    p_mesh_set = p_mesh.get(selected_pid, set())
    overlap_entities = sorted(q_ent & p_ent)
    overlap_mesh = sorted(q_mesh_set & p_mesh_set)
    question = questions.get(qid, {})
    passage = corpus.get(selected_pid, {})
    rank_gain = (baseline_rank_value - hgb_rank_value) if baseline_rank_value is not None else None
    return {
        "dataset": label,
        "question_id": qid,
        "passage_id": selected_pid,
        "rescued_gold_to_top_k": rescued,
        "baseline_gold_rank": baseline_rank_value,
        "hgb_gold_rank": hgb_rank_value,
        "rank_gain": rank_gain,
        "question": question.get("question", ""),
        "passage_title": passage.get("title", ""),
        "passage_text": passage.get("text", ""),
        "overlap_entities": overlap_entities[:12],
        "overlap_mesh": overlap_mesh[:12],
        "question_entities": sorted(q_ent)[:12],
        "passage_entities": sorted(p_ent)[:12],
        "question_mesh": sorted(q_mesh_set)[:12],
        "passage_mesh": sorted(p_mesh_set)[:12],
        "features": feature_values(hgb_row),
        "hybrid_score": base_row.get("score") if base_row else None,
        "hgb_score": hgb_row.get("score") if hgb_row else None,
    }


def collect_dataset_cases(label: str, prefix: str, top_k: int, max_cases: int) -> dict[str, Any]:
    paths = default_paths(prefix)
    questions = index_by_id(read_jsonl(paths["questions"]), "question_id")
    corpus = index_by_id(read_jsonl(paths["corpus"]), "passage_id")
    qrels_by_qid = group_qrels(read_jsonl(paths["qrels"]))
    baseline_by_qid = group_predictions(read_jsonl(paths["baseline"]))
    candidate_by_qid = group_predictions(read_jsonl(paths["candidate"]))
    baseline_rank = rank_map(baseline_by_qid)
    candidate_rank = rank_map(candidate_by_qid)

    q_entities = entity_map(read_jsonl(paths["question_entities"]), "question_id")
    p_entities = entity_map(read_jsonl(paths["passage_entities"]), "passage_id")
    q_mesh = mesh_map(read_jsonl(paths["question_mesh"]), "question_id")
    p_mesh = mesh_map(read_jsonl(paths["passage_mesh"]), "passage_id")

    qids = sorted(set(qrels_by_qid) & set(candidate_by_qid))
    summaries = {
        "num_questions": len(qids),
        "questions_with_rescued_gold_to_top_k": 0,
        "rescued_gold_passages_to_top_k": 0,
        "questions_with_lost_gold_from_top_k": 0,
        "lost_gold_passages_from_top_k": 0,
        "questions_with_gold_in_hgb_top100": 0,
        "questions_without_gold_in_hgb_top100": 0,
        "improved_min_gold_rank": 0,
        "unchanged_min_gold_rank": 0,
        "worsened_min_gold_rank": 0,
    }
    candidates: list[dict[str, Any]] = []

    for qid in qids:
        gold_ids = set(qrels_by_qid[qid])
        base_rank, base_pid = min_gold_rank(baseline_rank.get(qid, {}), gold_ids)
        hgb_rank, hgb_pid = min_gold_rank(candidate_rank.get(qid, {}), gold_ids)
        if hgb_rank is None:
            summaries["questions_without_gold_in_hgb_top100"] += 1
        else:
            summaries["questions_with_gold_in_hgb_top100"] += 1
            base_rank_for_compare = base_rank if base_rank is not None else 10**9
            if hgb_rank < base_rank_for_compare:
                summaries["improved_min_gold_rank"] += 1
            elif hgb_rank == base_rank_for_compare:
                summaries["unchanged_min_gold_rank"] += 1
            else:
                summaries["worsened_min_gold_rank"] += 1
        question_has_rescued_gold = False
        question_has_lost_gold = False

        for gold_pid in gold_ids:
            gold_base_row = baseline_rank.get(qid, {}).get(gold_pid)
            gold_hgb_row = candidate_rank.get(qid, {}).get(gold_pid)
            gold_base_rank = int(gold_base_row["rank"]) if gold_base_row else None
            gold_hgb_rank = int(gold_hgb_row["rank"]) if gold_hgb_row else None
            gold_base_rank_for_compare = gold_base_rank if gold_base_rank is not None else 10**9
            gold_hgb_rank_for_compare = gold_hgb_rank if gold_hgb_rank is not None else 10**9
            evidence_rescued = gold_base_rank_for_compare > top_k and gold_hgb_rank_for_compare <= top_k
            evidence_lost = gold_base_rank_for_compare <= top_k and gold_hgb_rank_for_compare > top_k
            if evidence_rescued:
                summaries["rescued_gold_passages_to_top_k"] += 1
                question_has_rescued_gold = True
            if evidence_lost:
                summaries["lost_gold_passages_from_top_k"] += 1
                question_has_lost_gold = True
            if gold_hgb_rank is None:
                continue
            if not evidence_rescued and gold_hgb_rank >= gold_base_rank_for_compare:
                continue
            candidates.append(
                build_case(
                    label=label,
                    qid=qid,
                    selected_pid=gold_pid,
                    baseline_rank_value=gold_base_rank,
                    hgb_rank_value=gold_hgb_rank,
                    rescued=evidence_rescued,
                    questions=questions,
                    corpus=corpus,
                    candidate_rank=candidate_rank,
                    baseline_rank=baseline_rank,
                    q_entities=q_entities,
                    p_entities=p_entities,
                    q_mesh=q_mesh,
                    p_mesh=p_mesh,
                )
            )

        if question_has_rescued_gold:
            summaries["questions_with_rescued_gold_to_top_k"] += 1
        if question_has_lost_gold:
            summaries["questions_with_lost_gold_from_top_k"] += 1

    def case_sort_key(item: dict[str, Any]) -> tuple[bool, float, int, str]:
        rank_gain = item["rank_gain"]
        sort_gain = 1000.0 if item["rescued_gold_to_top_k"] and rank_gain is None else float(rank_gain or 0.0)
        return (not item["rescued_gold_to_top_k"], -sort_gain, int(item["hgb_gold_rank"]), str(item["question_id"]))

    candidates.sort(key=case_sort_key)
    return {
        "dataset": label,
        "paths": paths,
        "summary": summaries,
        "cases": candidates[:max_cases],
    }


def write_markdown(path: str | Path, payload: dict[str, Any]) -> None:
    lines = ["# HGB Reranking Case Studies", ""]
    lines.append(
        f"Selection rule: prefer gold evidence moved from outside top-{payload['top_k']} "
        f"by Hybrid RRF into top-{payload['top_k']} by HGB; otherwise use the largest positive gold-rank gain."
    )
    lines.append("")
    for dataset in payload["datasets"]:
        summary = dataset["summary"]
        lines.append(f"## {dataset['dataset']}")
        lines.append("")
        lines.append(
            f"Questions analyzed: {summary['num_questions']}; rescued to top-{payload['top_k']}: "
            f"{summary['questions_with_rescued_gold_to_top_k']} questions / "
            f"{summary['rescued_gold_passages_to_top_k']} gold passages; lost from top-{payload['top_k']}: "
            f"{summary['questions_with_lost_gold_from_top_k']} questions / "
            f"{summary['lost_gold_passages_from_top_k']} gold passages; improved min-gold-rank: "
            f"{summary['improved_min_gold_rank']}; worsened: {summary['worsened_min_gold_rank']}; "
            f"no gold in HGB top100: {summary['questions_without_gold_in_hgb_top100']}."
        )
        lines.append("")
        lines.append(
            "| question_id | passage_id | Hybrid gold rank | HGB gold rank | rank gain | entity overlap | MeSH overlap | key HGB features |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | --- | --- | --- |")
        for case in dataset["cases"]:
            features = case["features"]
            feature_text = ", ".join(
                [
                    f"hypergraph={features.get('hypergraph_score_norm', 0.0):.3f}",
                    f"entity_cov={features.get('question_entity_coverage', 0.0):.3f}",
                    f"mesh_cov={features.get('question_mesh_coverage', 0.0):.3f}",
                    f"shared_edges={features.get('local_shared_entity_edges', 0.0):.0f}",
                    f"mesh_edges={features.get('local_document_mesh_edges', 0.0):.0f}",
                ]
            )
            base_rank = case["baseline_gold_rank"] if case["baseline_gold_rank"] is not None else ">100"
            rank_gain = case["rank_gain"] if case["rank_gain"] is not None else "n/a"
            lines.append(
                "| "
                + " | ".join(
                    [
                        escape_md(case["question_id"]),
                        escape_md(case["passage_id"]),
                        str(base_rank),
                        str(case["hgb_gold_rank"]),
                        str(rank_gain),
                        escape_md(", ".join(case["overlap_entities"]) or "none"),
                        escape_md(", ".join(case["overlap_mesh"]) or "none"),
                        escape_md(feature_text),
                    ]
                )
                + " |"
            )
        lines.append("")
        for idx, case in enumerate(dataset["cases"], start=1):
            lines.append(f"### {dataset['dataset']} case {idx}: {case['question_id']}")
            lines.append("")
            lines.append(f"Question: {escape_md(case['question'])}")
            lines.append("")
            title = f"{case['passage_title']} " if case["passage_title"] else ""
            lines.append(f"Gold passage: {escape_md(truncate(title + case['passage_text']))}")
            lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    datasets = [
        collect_dataset_cases(label, prefix, args.top_k, args.max_cases_per_dataset)
        for label, prefix in args.dataset
    ]
    payload = {
        "top_k": args.top_k,
        "max_cases_per_dataset": args.max_cases_per_dataset,
        "datasets": datasets,
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(
        {
            "output_json": args.output_json,
            "output_md": args.output_md,
            "datasets": [
                {
                    "dataset": item["dataset"],
                    "summary": item["summary"],
                    "num_cases": len(item["cases"]),
                }
                for item in datasets
            ],
        }
    )


if __name__ == "__main__":
    main()
