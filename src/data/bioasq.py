from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Any


ID_KEYS = ("id", "_id", "doc_id", "document_id", "passage_id", "pmid")
TEXT_KEYS = ("text", "passage", "context", "contents", "content", "abstract")
TITLE_KEYS = ("title", "name")
QUESTION_KEYS = ("question", "query", "question_text")
ANSWER_KEYS = ("answer", "answers", "ideal_answer", "exact_answer")
GOLD_KEYS = (
    "relevant_passage_ids",
    "relevant_passages",
    "passage_ids",
    "contexts",
    "context_ids",
    "documents",
    "document_ids",
)


def choose_split(dataset_dict: Any, requested_split: str | None = None) -> str:
    splits = list(dataset_dict.keys())
    if requested_split:
        if requested_split not in dataset_dict:
            raise ValueError(f"Requested split '{requested_split}' not found. Available: {splits}")
        return requested_split
    for split in ("train", "test", "validation", "dev"):
        if split in dataset_dict:
            return split
    if not splits:
        raise ValueError("Dataset has no available splits.")
    return splits[0]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return " ".join(normalize_text(v) for v in value if v is not None).strip()
    if isinstance(value, dict):
        return " ".join(normalize_text(v) for v in value.values() if v is not None).strip()
    return str(value).strip()


def first_present(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def infer_record_id(row: dict[str, Any], idx: int, prefix: str) -> str:
    value = first_present(row, ID_KEYS)
    if value is None:
        return f"{prefix}_{idx}"
    return str(value)


def infer_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    value = first_present(row, keys)
    if value is not None:
        return normalize_text(value)
    string_values = [normalize_text(v) for v in row.values() if isinstance(v, str)]
    return " ".join(v for v in string_values if v).strip()


def normalize_corpus(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, row in enumerate(rows):
        passage_id = infer_record_id(row, idx, "p")
        if passage_id in seen_ids:
            passage_id = f"{passage_id}_{idx}"
        seen_ids.add(passage_id)

        text = infer_text(row, TEXT_KEYS)
        if not text:
            continue
        title = normalize_text(first_present(row, TITLE_KEYS))
        records.append(
            {
                "passage_id": passage_id,
                "title": title,
                "text": text,
                "metadata": {"source_index": idx},
            }
        )
    return records


def normalize_answer(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(v) for v in value if normalize_text(v)]
    text = normalize_text(value)
    return [text] if text else []


def flatten_gold_ids(value: Any) -> list[str]:
    ids: list[str] = []
    if value is None:
        return ids
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        if text.startswith("[") or text.startswith("{"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if parsed is not None:
                return flatten_gold_ids(parsed)
        return [text] if text else []
    if isinstance(value, list):
        for item in value:
            ids.extend(flatten_gold_ids(item))
        return ids
    if isinstance(value, dict):
        for key in ("id", "passage_id", "doc_id", "document_id", "pmid"):
            if key in value and value[key] not in (None, ""):
                ids.append(str(value[key]))
                return ids
        for item in value.values():
            ids.extend(flatten_gold_ids(item))
    return ids


def map_gold_texts_to_ids(value: Any, text_to_passage_id: dict[str, str]) -> list[str]:
    mapped: list[str] = []
    if value is None:
        return mapped
    if isinstance(value, str):
        key = normalize_text(value).lower()
        if key in text_to_passage_id:
            mapped.append(text_to_passage_id[key])
        return mapped
    if isinstance(value, list):
        for item in value:
            mapped.extend(map_gold_texts_to_ids(item, text_to_passage_id))
    if isinstance(value, dict):
        for item in value.values():
            mapped.extend(map_gold_texts_to_ids(item, text_to_passage_id))
    return mapped


def normalize_questions(
    rows: Iterable[dict[str, Any]],
    corpus: list[dict[str, Any]],
    sample_size: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text_to_passage_id = {record["text"].lower(): record["passage_id"] for record in corpus}
    questions: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        if sample_size is not None and len(questions) >= sample_size:
            break
        question = infer_text(row, QUESTION_KEYS)
        if not question:
            continue

        question_id = infer_record_id(row, idx, "q")
        answers = normalize_answer(first_present(row, ANSWER_KEYS))

        gold_ids: list[str] = []
        for key in GOLD_KEYS:
            if key in row:
                gold_ids.extend(flatten_gold_ids(row[key]))
                gold_ids.extend(map_gold_texts_to_ids(row[key], text_to_passage_id))

        # Some versions expose passage ids nested inside a generic "passages" column.
        if "passages" in row:
            gold_ids.extend(flatten_gold_ids(row["passages"]))
            gold_ids.extend(map_gold_texts_to_ids(row["passages"], text_to_passage_id))

        deduped_gold_ids = sorted({pid for pid in gold_ids if pid})
        questions.append(
            {
                "question_id": question_id,
                "question": question,
                "answers": answers,
                "metadata": {"source_index": idx},
            }
        )
        for passage_id in deduped_gold_ids:
            qrels.append({"question_id": question_id, "passage_id": passage_id, "relevance": 1})

    return questions, qrels


def dataset_schema(rows: Iterable[dict[str, Any]], limit: int = 3) -> dict[str, Any]:
    preview = []
    columns: set[str] = set()
    for idx, row in enumerate(rows):
        columns.update(row.keys())
        if idx < limit:
            preview.append({k: type(v).__name__ for k, v in row.items()})
    return {"columns": sorted(columns), "preview_types": preview}
