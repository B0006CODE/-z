from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.entities import normalize_text
from src.utils import load_config, read_jsonl, write_json, write_jsonl


GENERIC_MESH = {
    "adolescent",
    "adult",
    "aged",
    "animals",
    "case-control studies",
    "cell line",
    "cells, cultured",
    "child",
    "child, preschool",
    "female",
    "humans",
    "infant",
    "male",
    "mice",
    "middle aged",
    "prospective studies",
    "rats",
    "retrospective studies",
    "time factors",
    "treatment outcome",
    "young adult",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build passage and question MeSH feature files from PubMed metadata.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--pubmed-mesh", default=None)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--passage-output", default=None)
    parser.add_argument("--question-output", default=None)
    parser.add_argument("--min-descriptor-frequency", type=int, default=2)
    parser.add_argument("--max-question-matches", type=int, default=32)
    parser.add_argument("--include-generic", action="store_true")
    return parser.parse_args()


def keep_descriptor(name: str, include_generic: bool) -> bool:
    normalized = normalize_text(name)
    if not normalized or len(normalized) < 3:
        return False
    if not include_generic and normalized in GENERIC_MESH:
        return False
    return True


def mesh_records(rows: list[dict[str, Any]], include_generic: bool) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    passage_rows = []
    descriptor_lookup: dict[str, dict[str, Any]] = {}
    descriptor_counts: Counter[str] = Counter()

    for row in rows:
        terms = []
        seen = set()
        for term in row.get("mesh_terms", []):
            ui = str(term.get("descriptor_ui", "")).strip()
            name = str(term.get("descriptor_name", "")).strip()
            if not ui or ui in seen or not keep_descriptor(name, include_generic):
                continue
            seen.add(ui)
            normalized = normalize_text(name)
            item = {
                "mesh_ui": ui,
                "mesh_name": name,
                "normalized": normalized,
                "major_topic": bool(term.get("major_topic", False)),
            }
            terms.append(item)
            descriptor_lookup.setdefault(ui, item)
            descriptor_counts[ui] += 1
        passage_rows.append({"passage_id": str(row["pmid"]), "mesh_terms": terms, "num_mesh_terms": len(terms)})

    filtered_lookup = {
        ui: item
        for ui, item in descriptor_lookup.items()
        if descriptor_counts[ui] >= 1
    }
    return passage_rows, filtered_lookup


def question_matches(
    question: str,
    descriptors: list[dict[str, Any]],
    *,
    max_matches: int,
) -> list[dict[str, Any]]:
    normalized_question = normalize_text(question)
    question_tokens = set(normalized_question.split())
    matches = []
    seen = set()
    for item in descriptors:
        name_norm = item["normalized"]
        if item["mesh_ui"] in seen:
            continue
        tokens = name_norm.split()
        if not tokens:
            continue
        exact = name_norm in normalized_question
        token_contained = 2 <= len(tokens) <= 5 and all(token in question_tokens for token in tokens)
        if exact or token_contained:
            seen.add(item["mesh_ui"])
            matches.append(
                {
                    "mesh_ui": item["mesh_ui"],
                    "mesh_name": item["mesh_name"],
                    "normalized": name_norm,
                    "match_type": "exact" if exact else "token_set",
                }
            )
    matches.sort(key=lambda row: (row["match_type"] != "exact", -len(row["normalized"]), row["mesh_name"]))
    return matches[:max_matches]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    pubmed_mesh_path = args.pubmed_mesh or paths.get("pubmed_mesh", "data/external_knowledge/pubmed_mesh.jsonl")
    questions_path = args.questions or paths["questions"]
    passage_output = args.passage_output or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")
    question_output = args.question_output or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")

    pubmed_rows = read_jsonl(pubmed_mesh_path)
    passage_rows, descriptor_lookup = mesh_records(pubmed_rows, include_generic=args.include_generic)
    descriptor_counts = Counter()
    for row in passage_rows:
        for term in row["mesh_terms"]:
            descriptor_counts[term["mesh_ui"]] += 1
    descriptors = [
        item
        for ui, item in descriptor_lookup.items()
        if descriptor_counts[ui] >= args.min_descriptor_frequency
    ]
    descriptors.sort(key=lambda row: (-len(row["normalized"]), row["mesh_name"]))

    question_rows = []
    for row in read_jsonl(questions_path):
        matches = question_matches(str(row["question"]), descriptors, max_matches=args.max_question_matches)
        question_rows.append(
            {
                "question_id": str(row["question_id"]),
                "mesh_terms": matches,
                "num_mesh_terms": len(matches),
            }
        )

    write_jsonl(passage_output, passage_rows)
    write_jsonl(question_output, question_rows)
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "pubmed_mesh": pubmed_mesh_path,
        "questions": questions_path,
        "passage_output": passage_output,
        "question_output": question_output,
        "num_pubmed_records": len(pubmed_rows),
        "num_passage_records": len(passage_rows),
        "num_descriptors": len(descriptor_lookup),
        "num_descriptors_after_frequency_filter": len(descriptors),
        "min_descriptor_frequency": args.min_descriptor_frequency,
        "include_generic": args.include_generic,
        "passages_with_mesh": sum(1 for row in passage_rows if row["num_mesh_terms"] > 0),
        "avg_mesh_per_passage": sum(row["num_mesh_terms"] for row in passage_rows) / len(passage_rows) if passage_rows else 0.0,
        "questions_with_mesh": sum(1 for row in question_rows if row["num_mesh_terms"] > 0),
        "avg_mesh_per_question": sum(row["num_mesh_terms"] for row in question_rows) / len(question_rows) if question_rows else 0.0,
    }
    write_json(Path(paths["logs_dir"]) / "build_mesh_features_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
