from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.entities import normalize_text
from src.retrieval.bm25 import BM25Retriever
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run field-aware BM25 with MeSH query and document expansion.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--mesh-synonyms", default=None)
    parser.add_argument("--output", default="outputs/retrieval/fielded_bm25_full_top100.jsonl")
    parser.add_argument("--index-path", default="indexes/bm25/bioasq_fielded_bm25.pkl")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--title-weight", type=int, default=3)
    parser.add_argument("--mesh-weight", type=int, default=2)
    parser.add_argument("--major-mesh-weight", type=int, default=4)
    parser.add_argument("--query-mesh-weight", type=int, default=2)
    parser.add_argument("--max-mesh-terms", type=int, default=24)
    parser.add_argument("--max-synonyms-per-term", type=int, default=2)
    return parser.parse_args()


def rows_by_id(rows: list[dict[str, Any]], id_key: str) -> dict[str, dict[str, Any]]:
    return {str(row[id_key]): row for row in rows}


def load_synonym_lookup(path: str | None) -> dict[str, list[str]]:
    if not path or not Path(path).exists():
        return {}
    lookup: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        values = []
        for value in [row.get("mesh_name", ""), *row.get("entry_terms", [])]:
            normalized = normalize_text(str(value))
            if normalized and normalized not in values:
                values.append(normalized)
        lookup[str(row["mesh_ui"])] = values
    return lookup


def repeated(text: str, weight: int) -> str:
    value = text.strip()
    if not value or weight <= 0:
        return ""
    return " ".join([value] * weight)


def mesh_expansion_text(
    mesh_terms: list[dict[str, Any]],
    synonym_lookup: dict[str, list[str]],
    *,
    mesh_weight: int,
    major_mesh_weight: int,
    max_mesh_terms: int,
    max_synonyms_per_term: int,
) -> str:
    parts: list[str] = []
    for term in mesh_terms[:max_mesh_terms]:
        mesh_ui = str(term.get("mesh_ui", ""))
        base = str(term.get("normalized") or normalize_text(str(term.get("mesh_name", ""))))
        weight = major_mesh_weight if term.get("major_topic") else mesh_weight
        parts.append(repeated(base, weight))
        for synonym in synonym_lookup.get(mesh_ui, [])[:max_synonyms_per_term]:
            if synonym != base:
                parts.append(synonym)
    return " ".join(part for part in parts if part)


def augment_corpus(
    corpus: list[dict[str, Any]],
    passage_mesh_by_id: dict[str, dict[str, Any]],
    synonym_lookup: dict[str, list[str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    augmented = []
    for row in corpus:
        pid = str(row["passage_id"])
        mesh_terms = passage_mesh_by_id.get(pid, {}).get("mesh_terms", [])
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))
        field_text = " ".join(
            part
            for part in [
                repeated(title, args.title_weight),
                text,
                mesh_expansion_text(
                    mesh_terms,
                    synonym_lookup,
                    mesh_weight=args.mesh_weight,
                    major_mesh_weight=args.major_mesh_weight,
                    max_mesh_terms=args.max_mesh_terms,
                    max_synonyms_per_term=args.max_synonyms_per_term,
                ),
            ]
            if part
        )
        new_row = dict(row)
        new_row["text"] = field_text
        augmented.append(new_row)
    return augmented


def expanded_query(
    question: dict[str, Any],
    question_mesh_by_id: dict[str, dict[str, Any]],
    synonym_lookup: dict[str, list[str]],
    args: argparse.Namespace,
) -> str:
    qid = str(question["question_id"])
    mesh_terms = question_mesh_by_id.get(qid, {}).get("mesh_terms", [])
    expansion = mesh_expansion_text(
        mesh_terms,
        synonym_lookup,
        mesh_weight=args.query_mesh_weight,
        major_mesh_weight=args.query_mesh_weight,
        max_mesh_terms=args.max_mesh_terms,
        max_synonyms_per_term=args.max_synonyms_per_term,
    )
    return " ".join(part for part in [str(question["question"]), expansion] if part)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    paths = config["paths"]
    top_k = args.top_k or int(config["retrieval"].get("top_k", 100))
    corpus_path = args.corpus or paths["corpus"]
    questions_path = args.questions or paths["questions"]
    question_mesh_path = args.question_mesh or paths.get("question_mesh")
    passage_mesh_path = args.passage_mesh or paths.get("passage_mesh")
    mesh_synonyms_path = args.mesh_synonyms or paths.get("mesh_synonyms")

    corpus = read_jsonl(corpus_path)
    questions = read_jsonl(questions_path)
    if args.sample_limit is not None:
        questions = questions[: args.sample_limit]
    question_mesh_by_id = rows_by_id(read_jsonl(question_mesh_path), "question_id") if question_mesh_path else {}
    passage_mesh_by_id = rows_by_id(read_jsonl(passage_mesh_path), "passage_id") if passage_mesh_path else {}
    synonym_lookup = load_synonym_lookup(mesh_synonyms_path)

    index_file = Path(args.index_path)
    if index_file.exists() and not args.rebuild_index:
        retriever = BM25Retriever.load(index_file)
        index_action = "loaded"
    else:
        retriever = BM25Retriever(
            k1=float(config["retrieval"].get("bm25", {}).get("k1", 1.5)),
            b=float(config["retrieval"].get("bm25", {}).get("b", 0.75)),
        )
        retriever.fit(augment_corpus(corpus, passage_mesh_by_id, synonym_lookup, args))
        retriever.save(index_file)
        index_action = "built"

    predictions = []
    expanded_question_count = 0
    for question in questions:
        query = expanded_query(question, question_mesh_by_id, synonym_lookup, args)
        if query.strip() != str(question["question"]).strip():
            expanded_question_count += 1
        for result in retriever.search(query, top_k=top_k):
            predictions.append(
                {
                    "question_id": question["question_id"],
                    "passage_id": result["passage_id"],
                    "rank": result["rank"],
                    "score": result["score"],
                    "retriever": "fielded_bm25_mesh",
                    "metadata": {
                        "top_k": top_k,
                        "title_weight": args.title_weight,
                        "mesh_weight": args.mesh_weight,
                        "major_mesh_weight": args.major_mesh_weight,
                        "query_mesh_weight": args.query_mesh_weight,
                    },
                }
            )
    write_jsonl(args.output, predictions)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "index_action": index_action,
        "index_path": str(index_file),
        "questions": questions_path,
        "corpus": corpus_path,
        "question_mesh": question_mesh_path,
        "passage_mesh": passage_mesh_path,
        "mesh_synonyms": mesh_synonyms_path,
        "num_questions": len(questions),
        "num_corpus_passages": len(corpus),
        "questions_with_mesh_expansion": expanded_question_count,
        "passages_with_mesh": sum(1 for row in passage_mesh_by_id.values() if row.get("mesh_terms")),
        "top_k": top_k,
        "num_predictions": len(predictions),
        "output": args.output,
    }
    write_json(Path(paths["logs_dir"]) / "run_fielded_bm25_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
