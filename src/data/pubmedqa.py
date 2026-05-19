from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable

from src.data.bioasq import normalize_text
from src.knowledge.entities import normalize_text as normalize_concept


GENERIC_MESH = {
    "adolescent",
    "adult",
    "aged",
    "animals",
    "case-control studies",
    "cell line",
    "cells, cultured",
    "child",
    "female",
    "humans",
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


def stable_mesh_ui(term: str) -> str:
    digest = hashlib.sha1(normalize_concept(term).encode("utf-8")).hexdigest()[:12]
    return f"PMQA_MESH_{digest}"


def mesh_item(term: str) -> dict[str, Any] | None:
    name = normalize_text(term)
    normalized = normalize_concept(name)
    if not name or len(normalized) < 3 or normalized in GENERIC_MESH:
        return None
    return {
        "mesh_ui": stable_mesh_ui(name),
        "mesh_name": name,
        "normalized": normalized,
        "major_topic": False,
    }


def context_parts(row: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    context = row.get("context") or {}
    if not isinstance(context, dict):
        return [], [], []
    texts = [normalize_text(item) for item in context.get("contexts", [])]
    labels = [normalize_text(item) for item in context.get("labels", [])]
    meshes = [normalize_text(item) for item in context.get("meshes", [])]
    return texts, labels, meshes


def normalize_pubmedqa(
    rows: Iterable[dict[str, Any]],
    *,
    sample_size: int | None = None,
    question_prefix: str = "pubmedqa",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    questions: list[dict[str, Any]] = []
    corpus: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    passage_mesh_rows: list[dict[str, Any]] = []
    source_rows = list(rows)

    for source_index, row in enumerate(source_rows):
        if sample_size is not None and len(questions) >= sample_size:
            break
        question = normalize_text(row.get("question"))
        pubid = str(row.get("pubid", f"row_{source_index}"))
        if not question:
            continue
        contexts, context_labels, meshes = context_parts(row)
        if not contexts:
            continue

        question_id = f"{question_prefix}_{pubid}"
        question_mesh_terms = [item for term in meshes if (item := mesh_item(term)) is not None]
        mesh_terms = list({term["mesh_ui"]: term for term in question_mesh_terms}.values())

        questions.append(
            {
                "question_id": question_id,
                "question": question,
                "answers": [normalize_text(row.get("long_answer"))] if normalize_text(row.get("long_answer")) else [],
                "metadata": {
                    "source_index": source_index,
                    "dataset": "qiaojin/PubMedQA",
                    "config": "pqa_labeled",
                    "pubid": pubid,
                    "final_decision": normalize_text(row.get("final_decision")).lower(),
                },
            }
        )
        labels.append(
            {
                "question_id": question_id,
                "pubid": pubid,
                "final_decision": normalize_text(row.get("final_decision")).lower(),
                "long_answer": normalize_text(row.get("long_answer")),
            }
        )

        for context_index, text in enumerate(contexts):
            if not text:
                continue
            section_label = context_labels[context_index] if context_index < len(context_labels) else ""
            passage_id = f"pubmedqa_{pubid}_{context_index}"
            corpus.append(
                {
                    "passage_id": passage_id,
                    "title": f"PMID {pubid} {section_label}".strip(),
                    "text": text,
                    "metadata": {
                        "source_index": source_index,
                        "pubid": pubid,
                        "context_index": context_index,
                        "section_label": section_label,
                        "mesh_terms": mesh_terms,
                    },
                }
            )
            qrels.append({"question_id": question_id, "passage_id": passage_id, "relevance": 1})
            passage_mesh_rows.append(
                {
                    "passage_id": passage_id,
                    "mesh_terms": mesh_terms,
                    "num_mesh_terms": len(mesh_terms),
                }
            )

    # The question rows intentionally do not carry all MeSH terms in metadata because that
    # would leak abstract-level labels into the text input. Build lexical question matches below.
    question_mesh_rows = build_question_mesh_rows(questions, passage_mesh_rows)
    return questions, corpus, qrels, labels, passage_mesh_rows, question_mesh_rows


def build_question_mesh_rows(
    questions: list[dict[str, Any]],
    passage_mesh_rows: list[dict[str, Any]],
    *,
    min_frequency: int = 1,
    max_matches: int = 32,
) -> list[dict[str, Any]]:
    descriptor_by_ui: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for row in passage_mesh_rows:
        for term in row.get("mesh_terms", []):
            ui = str(term.get("mesh_ui", ""))
            if not ui:
                continue
            descriptor_by_ui[ui] = term
            counts[ui] += 1
    descriptors = [
        term
        for ui, term in descriptor_by_ui.items()
        if counts[ui] >= min_frequency
    ]
    descriptors.sort(key=lambda term: (-len(str(term.get("normalized", ""))), str(term.get("mesh_name", ""))))

    records = []
    for question in questions:
        normalized_question = normalize_concept(question["question"])
        question_tokens = set(normalized_question.split())
        matches = []
        seen: set[str] = set()
        for term in descriptors:
            ui = str(term.get("mesh_ui", ""))
            normalized = str(term.get("normalized", ""))
            if not ui or ui in seen or not normalized:
                continue
            tokens = normalized.split()
            exact = normalized in normalized_question
            token_contained = 2 <= len(tokens) <= 5 and all(token in question_tokens for token in tokens)
            if exact or token_contained:
                seen.add(ui)
                matches.append(
                    {
                        "mesh_ui": ui,
                        "mesh_name": term.get("mesh_name", ""),
                        "normalized": normalized,
                        "match_type": "exact" if exact else "token_set",
                    }
                )
        records.append(
            {
                "question_id": question["question_id"],
                "mesh_terms": matches[:max_matches],
                "num_mesh_terms": min(len(matches), max_matches),
            }
        )
    return records


def summarize_pubmedqa(
    questions: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    passage_mesh_rows: list[dict[str, Any]],
    question_mesh_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    label_counts = Counter(row.get("final_decision", "") for row in labels)
    return {
        "num_questions": len(questions),
        "num_corpus_passages": len(corpus),
        "num_qrels": len(qrels),
        "num_questions_with_qrels": len({row["question_id"] for row in qrels}),
        "answer_label_counts": dict(sorted(label_counts.items())),
        "passages_with_mesh": sum(1 for row in passage_mesh_rows if row.get("num_mesh_terms", 0) > 0),
        "questions_with_mesh": sum(1 for row in question_mesh_rows if row.get("num_mesh_terms", 0) > 0),
        "avg_qrels_per_question": len(qrels) / len(questions) if questions else 0.0,
        "avg_passage_mesh_terms": (
            sum(row.get("num_mesh_terms", 0) for row in passage_mesh_rows) / len(passage_mesh_rows)
            if passage_mesh_rows
            else 0.0
        ),
        "avg_question_mesh_terms": (
            sum(row.get("num_mesh_terms", 0) for row in question_mesh_rows) / len(question_mesh_rows)
            if question_mesh_rows
            else 0.0
        ),
    }
