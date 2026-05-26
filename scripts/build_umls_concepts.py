from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.entities import normalize_text, tokenize
from src.utils import load_config, read_jsonl, write_json, write_jsonl


MRCONSO_FIELDS = [
    "CUI",
    "LAT",
    "TS",
    "LUI",
    "STT",
    "SUI",
    "ISPREF",
    "AUI",
    "SAUI",
    "SCUI",
    "SDUI",
    "SAB",
    "TTY",
    "CODE",
    "STR",
    "SRL",
    "SUPPRESS",
    "CVF",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lightweight UMLS CUI annotations for project questions/passages.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mrconso", required=True, help="Path to UMLS MRCONSO.RRF.")
    parser.add_argument("--mrsty", default=None, help="Optional path to UMLS MRSTY.RRF.")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--question-output", default="data/processed/bioasq_question_cui.jsonl")
    parser.add_argument("--passage-output", default="data/processed/bioasq_passage_cui.jsonl")
    parser.add_argument("--stats-output", default="results/metrics/umls_concept_stats.json")
    parser.add_argument("--allowed-sabs", default="MSH,RXNORM,SNOMEDCT_US,MED-RT,NCI,OMIM,HPO,HGNC")
    parser.add_argument("--max-terms", type=int, default=750000)
    parser.add_argument("--max-term-tokens", type=int, default=8)
    parser.add_argument("--min-term-chars", type=int, default=3)
    parser.add_argument("--max-matches", type=int, default=128)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def read_mrsty(path: str | Path | None) -> dict[str, list[str]]:
    if not path:
        return {}
    sty_by_cui: dict[str, list[str]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4:
                continue
            cui = parts[0]
            sty = parts[3]
            if cui and sty and sty not in sty_by_cui[cui]:
                sty_by_cui[cui].append(sty)
    return dict(sty_by_cui)


def valid_umls_term(term: str, *, min_chars: int, max_tokens: int) -> bool:
    normalized = normalize_text(term)
    if len(normalized) < min_chars:
        return False
    tokens = normalized.split()
    if not tokens or len(tokens) > max_tokens:
        return False
    if len(tokens) == 1 and len(tokens[0]) < min_chars:
        return False
    if any(token.isdigit() and len(token) > 4 for token in tokens):
        return False
    return True


def build_umls_index(
    mrconso_path: str | Path,
    *,
    allowed_sabs: set[str],
    max_terms: int,
    min_chars: int,
    max_tokens: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, int]]:
    term_map: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    stats = defaultdict(int)
    with Path(mrconso_path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stats["mrconso_rows"] += 1
            parts = line.rstrip("\n").split("|")
            if len(parts) < len(MRCONSO_FIELDS):
                continue
            row = dict(zip(MRCONSO_FIELDS, parts, strict=False))
            if row["LAT"] != "ENG":
                continue
            if row["SUPPRESS"] not in {"N", "O"}:
                continue
            if allowed_sabs and row["SAB"] not in allowed_sabs:
                continue
            term = row["STR"].strip()
            if not valid_umls_term(term, min_chars=min_chars, max_tokens=max_tokens):
                continue
            normalized = normalize_text(term)
            cui = row["CUI"]
            if not cui or cui in term_map[normalized]:
                continue
            term_map[normalized][cui] = {
                "cui": cui,
                "term": term,
                "normalized": normalized,
                "sab": row["SAB"],
                "tty": row["TTY"],
                "code": row["CODE"],
            }
            stats["kept_terms"] += 1
            if stats["kept_terms"] >= max_terms:
                break
    first_token_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for normalized, cui_rows in term_map.items():
        tokens = normalized.split()
        if not tokens:
            continue
        for row in cui_rows.values():
            indexed = dict(row)
            indexed["tokens"] = " ".join(tokens)
            indexed["num_tokens"] = str(len(tokens))
            first_token_index[tokens[0]].append(indexed)
    for rows in first_token_index.values():
        rows.sort(key=lambda item: (-int(item["num_tokens"]), item["normalized"], item["cui"]))
    stats["unique_normalized_terms"] = len(term_map)
    stats["first_tokens"] = len(first_token_index)
    return dict(first_token_index), dict(stats)


def match_umls_concepts(
    text: str,
    index: dict[str, list[dict[str, str]]],
    sty_by_cui: dict[str, list[str]],
    *,
    max_matches: int,
) -> list[dict[str, Any]]:
    tokens = [token.lower() for token in tokenize(text)]
    matches: list[dict[str, Any]] = []
    occupied: set[int] = set()
    seen: set[str] = set()
    for start, token in enumerate(tokens):
        if start in occupied:
            continue
        for entry in index.get(token, []):
            term_tokens = str(entry["tokens"]).split()
            end = start + len(term_tokens)
            if end > len(tokens):
                continue
            if tokens[start:end] != term_tokens:
                continue
            cui = entry["cui"]
            if cui in seen:
                continue
            if any(pos in occupied for pos in range(start, end)):
                continue
            seen.add(cui)
            occupied.update(range(start, end))
            matches.append(
                {
                    "cui": cui,
                    "term": entry["term"],
                    "normalized": entry["normalized"],
                    "sab": entry["sab"],
                    "tty": entry["tty"],
                    "code": entry["code"],
                    "semantic_types": sty_by_cui.get(cui, []),
                    "start_token": start,
                    "end_token": end,
                }
            )
            break
        if len(matches) >= max_matches:
            break
    return matches


def annotate_records(
    records: list[dict[str, Any]],
    *,
    id_key: str,
    text_keys: list[str],
    index: dict[str, list[dict[str, str]]],
    sty_by_cui: dict[str, list[str]],
    max_matches: int,
    max_records: int | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(records):
        if max_records is not None and idx >= max_records:
            break
        text = " ".join(str(row.get(key, "")) for key in text_keys if row.get(key))
        concepts = match_umls_concepts(text, index, sty_by_cui, max_matches=max_matches)
        output.append(
            {
                id_key: str(row[id_key]),
                "cui_concepts": concepts,
                "num_cui_concepts": len(concepts),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    questions_path = args.questions or paths.get("questions", "data/processed/bioasq_questions.jsonl")
    corpus_path = args.corpus or paths.get("corpus", "data/processed/bioasq_corpus.jsonl")
    allowed_sabs = {item.strip() for item in args.allowed_sabs.split(",") if item.strip()}

    sty_by_cui = read_mrsty(args.mrsty)
    index, index_stats = build_umls_index(
        args.mrconso,
        allowed_sabs=allowed_sabs,
        max_terms=args.max_terms,
        min_chars=args.min_term_chars,
        max_tokens=args.max_term_tokens,
    )
    questions = read_jsonl(questions_path)
    corpus = read_jsonl(corpus_path)
    q_annotations = annotate_records(
        questions,
        id_key="question_id",
        text_keys=["question"],
        index=index,
        sty_by_cui=sty_by_cui,
        max_matches=args.max_matches,
        max_records=args.max_records,
    )
    p_annotations = annotate_records(
        corpus,
        id_key="passage_id",
        text_keys=["title", "text"],
        index=index,
        sty_by_cui=sty_by_cui,
        max_matches=args.max_matches,
        max_records=args.max_records,
    )
    write_jsonl(args.question_output, q_annotations)
    write_jsonl(args.passage_output, p_annotations)
    stats = {
        "timestamp": datetime.now(UTC).isoformat(),
        "mrconso": str(args.mrconso),
        "mrsty": str(args.mrsty) if args.mrsty else None,
        "allowed_sabs": sorted(allowed_sabs),
        "questions": questions_path,
        "corpus": corpus_path,
        "question_output": args.question_output,
        "passage_output": args.passage_output,
        "index_stats": index_stats,
        "num_question_records": len(q_annotations),
        "num_passage_records": len(p_annotations),
        "question_records_with_cui": sum(1 for row in q_annotations if row["num_cui_concepts"] > 0),
        "passage_records_with_cui": sum(1 for row in p_annotations if row["num_cui_concepts"] > 0),
        "mean_question_cui": sum(row["num_cui_concepts"] for row in q_annotations) / len(q_annotations) if q_annotations else 0.0,
        "mean_passage_cui": sum(row["num_cui_concepts"] for row in p_annotations) / len(p_annotations) if p_annotations else 0.0,
    }
    write_json(args.stats_output, stats)
    write_json(Path(paths.get("logs_dir", "logs")) / "build_umls_concepts_summary.json", stats)
    print(stats)


if __name__ == "__main__":
    main()
