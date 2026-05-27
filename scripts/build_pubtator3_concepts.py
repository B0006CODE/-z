from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_jsonl, write_json, write_jsonl


PUBTATOR3_EXPORT_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
SUPPORTED_TYPES = {"disease", "chemical", "gene", "species", "mutation", "variant", "cellline", "cell line"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PubTator3 concept annotations for BioASQ PMID passages.")
    parser.add_argument("--corpus", default="data/processed/bioasq_corpus.jsonl")
    parser.add_argument("--questions", default="data/processed/bioasq_questions.jsonl")
    parser.add_argument("--candidate-predictions", default="outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl")
    parser.add_argument("--max-qids", type=int, default=10)
    parser.add_argument("--candidate-top-k", type=int, default=30)
    parser.add_argument("--max-pmids", type=int, default=3000)
    parser.add_argument("--qid-selection-order", choices=["numeric", "lexical"], default="numeric")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=0.35)
    parser.add_argument("--cache-dir", default="data/external_knowledge/pubtator3_cache")
    parser.add_argument("--passage-output", default="data/processed/bioasq_passage_pubtator_concepts.jsonl")
    parser.add_argument("--question-output", default="data/processed/bioasq_question_pubtator_concepts.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/pubtator3_concept_coverage.json")
    return parser.parse_args()


def qid_sort_key(qid: str, order: str) -> tuple[int, int | str]:
    if order == "numeric" and qid.isdigit():
        return (0, int(qid))
    return (1, qid)


def selected_qids(predictions: list[dict[str, Any]], max_qids: int | None, order: str) -> set[str]:
    qids = sorted({str(row["question_id"]) for row in predictions}, key=lambda qid: qid_sort_key(qid, order))
    return set(qids[:max_qids]) if max_qids is not None else set(qids)


def pmid_like(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,9}", value.strip()))


def fetch_batch(pmids: list[str], cache_dir: Path, sleep_seconds: float) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / ("pmids_" + "_".join(pmids[:3]) + f"_n{len(pmids)}.json")
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    url = PUBTATOR3_EXPORT_URL + "?" + urllib.parse.urlencode({"pmids": ",".join(pmids)})
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    time.sleep(sleep_seconds)
    return payload


def normalize_type(raw_type: str) -> str:
    value = raw_type.strip().lower().replace("_", " ")
    if value == "variant":
        return "mutation"
    if value == "cell line":
        return "cellline"
    return value


def annotation_concept(annotation: dict[str, Any], *, pmid: str, passage_type: str) -> dict[str, Any] | None:
    infons = annotation.get("infons", {})
    concept_type = normalize_type(str(infons.get("type") or infons.get("biotype") or ""))
    if concept_type not in SUPPORTED_TYPES:
        return None
    normalized = infons.get("normalized") or []
    if isinstance(normalized, str):
        normalized = [normalized]
    concept_id = (
        str(infons.get("normalized_id") or "").strip()
        or str(infons.get("identifier") or "").strip()
        or (str(normalized[0]).strip() if normalized else "")
        or str(infons.get("accession") or "").strip()
    )
    text = str(annotation.get("text") or "").strip()
    name = str(infons.get("name") or text).strip()
    if not concept_id and not name:
        return None
    return {
        "concept_id": concept_id or name.lower(),
        "type": concept_type,
        "name": name,
        "mention": text,
        "pmid": pmid,
        "passage_type": passage_type,
        "database": infons.get("database"),
    }


def parse_pubtator3_payload(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_pmid: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = defaultdict(dict)
    for document in payload.get("PubTator3", []):
        pmid = str(document.get("id") or "").strip()
        if not pmid:
            continue
        for passage in document.get("passages", []):
            passage_type = str(passage.get("infons", {}).get("type") or "")
            for annotation in passage.get("annotations", []):
                concept = annotation_concept(annotation, pmid=pmid, passage_type=passage_type)
                if not concept:
                    continue
                key = (concept["type"], concept["concept_id"], concept["name"].lower())
                by_pmid[pmid][key] = concept
    return {pmid: sorted(concepts.values(), key=lambda item: (item["type"], item["concept_id"], item["name"])) for pmid, concepts in by_pmid.items()}


def fetch_pubtator_concepts(pmids: list[str], *, batch_size: int, cache_dir: Path, sleep_seconds: float) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    concepts_by_pmid: dict[str, list[dict[str, Any]]] = {}
    failed: list[str] = []
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        try:
            payload = fetch_batch(batch, cache_dir, sleep_seconds)
            concepts_by_pmid.update(parse_pubtator3_payload(payload))
        except Exception:
            failed.extend(batch)
    return concepts_by_pmid, failed


def concept_lexicon(concepts_by_pmid: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for concepts in concepts_by_pmid.values():
        for concept in concepts:
            names = {str(concept.get("name") or "").strip(), str(concept.get("mention") or "").strip()}
            for name in names:
                if len(name) < 3:
                    continue
                dedup[(concept["type"], name.lower())] = {**concept, "name": name}
    return sorted(dedup.values(), key=lambda item: (-len(item["name"]), item["type"], item["name"].lower()))


def annotate_questions_by_lexicon(questions: list[dict[str, Any]], lexicon: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in questions:
        qid = str(row["question_id"])
        text = str(row.get("question") or "")
        lowered = text.lower()
        found: dict[tuple[str, str], dict[str, Any]] = {}
        for concept in lexicon:
            name = str(concept["name"]).lower()
            if name and re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", lowered):
                key = (concept["type"], concept["concept_id"])
                found[key] = {
                    "concept_id": concept["concept_id"],
                    "type": concept["type"],
                    "name": concept["name"],
                    "mention": concept["name"],
                    "source": "pubtator3_lexicon_match",
                }
        rows.append({"question_id": qid, "concepts": sorted(found.values(), key=lambda item: (item["type"], item["concept_id"]))})
    return rows


def main() -> None:
    args = parse_args()
    predictions = read_jsonl(args.candidate_predictions)
    qids = selected_qids(predictions, args.max_qids, args.qid_selection_order)
    candidate_pmids = sorted(
        {
            str(row["passage_id"])
            for row in predictions
            if str(row["question_id"]) in qids and pmid_like(str(row["passage_id"]))
            and int(row.get("rank", args.candidate_top_k + 1)) <= args.candidate_top_k
        },
        key=lambda value: int(value),
    )
    corpus_ids = {str(row["passage_id"]) for row in read_jsonl(args.corpus)}
    candidate_pmids = [pmid for pmid in candidate_pmids if pmid in corpus_ids]
    if args.max_pmids is not None:
        candidate_pmids = candidate_pmids[: args.max_pmids]
    concepts_by_pmid, failed_pmids = fetch_pubtator_concepts(
        candidate_pmids,
        batch_size=args.batch_size,
        cache_dir=Path(args.cache_dir),
        sleep_seconds=args.sleep_seconds,
    )
    passage_rows = [
        {"passage_id": pmid, "pmid": pmid, "concepts": concepts_by_pmid.get(pmid, [])}
        for pmid in candidate_pmids
    ]
    questions = [row for row in read_jsonl(args.questions) if str(row["question_id"]) in qids]
    question_rows = annotate_questions_by_lexicon(questions, concept_lexicon(concepts_by_pmid))
    write_jsonl(args.passage_output, passage_rows)
    write_jsonl(args.question_output, question_rows)

    passage_type_counts: Counter[str] = Counter()
    for concepts in concepts_by_pmid.values():
        for concept in concepts:
            passage_type_counts[str(concept["type"])] += 1
    question_type_counts: Counter[str] = Counter()
    for row in question_rows:
        for concept in row["concepts"]:
            question_type_counts[str(concept["type"])] += 1
    metrics = {
        "timestamp": datetime.now(UTC).isoformat(),
        "api": PUBTATOR3_EXPORT_URL,
        "candidate_predictions": args.candidate_predictions,
        "max_qids": args.max_qids,
        "num_questions": len(qids),
        "num_candidate_pmids": len(candidate_pmids),
        "candidate_top_k": args.candidate_top_k,
        "max_pmids": args.max_pmids,
        "num_failed_pmids": len(failed_pmids),
        "failed_pmids_preview": failed_pmids[:20],
        "num_passages_with_concepts": sum(1 for row in passage_rows if row["concepts"]),
        "passage_concept_count": sum(len(row["concepts"]) for row in passage_rows),
        "passage_concept_type_counts": dict(passage_type_counts),
        "num_questions_with_concepts": sum(1 for row in question_rows if row["concepts"]),
        "question_concept_count": sum(len(row["concepts"]) for row in question_rows),
        "question_concept_type_counts": dict(question_type_counts),
        "question_annotation_note": "Questions are matched against names and mentions from PubTator3 PMID annotations in the selected candidate pool.",
        "passage_output": args.passage_output,
        "question_output": args.question_output,
    }
    write_json(args.metrics_output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
