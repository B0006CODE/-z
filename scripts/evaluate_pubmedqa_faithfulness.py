from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.pubmedqa_answering import group_predictions
from src.utils import load_config, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate rule-based PubMedQA answer-support diagnostics for generated/selected "
            "yes/no/maybe labels. This is not an LLM judge."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=QA_JSONL=EVIDENCE_JSONL",
        help="Repeatable source triple, for example hgb=outputs/generation/x.jsonl=outputs/rerank/y.jsonl.",
    )
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--output", default="results/metrics/pubmedqa_faithfulness_metrics.json")
    parser.add_argument("--details-output", default="outputs/generation/pubmedqa_faithfulness_details.jsonl")
    parser.add_argument("--table-output", default="results/tables/pubmedqa_faithfulness.md")
    return parser.parse_args()


def parse_source(value: str) -> tuple[str, str, str]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Expected NAME=QA_JSONL=EVIDENCE_JSONL for --source, got: {value}")
    return parts[0], parts[1], parts[2]


def qrels_map(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[str(row["question_id"])].add(str(row["passage_id"]))
    return dict(grouped)


def entity_map(rows: list[dict[str, Any]], id_key: str) -> dict[str, set[str]]:
    mapped: dict[str, set[str]] = {}
    for row in rows:
        ids = {
            str(entity.get("entity_id") or entity.get("canonical", "")).strip().lower()
            for entity in row.get("entities", [])
        }
        mapped[str(row[id_key])] = {entity_id for entity_id in ids if entity_id}
    return mapped


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    citation_supported = sum(1 for row in rows if row["citation_supported"])
    label_supported = sum(1 for row in rows if row["label_supported"])
    answer_supported = sum(1 for row in rows if row["answer_supported"])
    entity_evaluable = sum(1 for row in rows if row["entity_evaluable"])
    entity_consistent = sum(1 for row in rows if row["entity_consistent"])
    return {
        "num_eval": total,
        "citation_support_rate": safe_rate(citation_supported, total),
        "answer_label_accuracy": safe_rate(label_supported, total),
        "answer_supported_rate": safe_rate(answer_supported, total),
        "unsupported_claim_rate": 1.0 - safe_rate(answer_supported, total),
        "entity_evaluable": entity_evaluable,
        "answer_evidence_entity_consistency": safe_rate(entity_consistent, entity_evaluable),
    }


def markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| evidence_source | method | top_k | num_eval | citation_support | label_accuracy | answer_supported | unsupported_claim | entity_evaluable | entity_consistency |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        metrics = row["metrics"]
        lines.append(
            "| {source} | {method} | {top_k} | {num_eval} | {citation:.4f} | {label:.4f} | {supported:.4f} | {unsupported:.4f} | {entity_n} | {entity:.4f} |".format(
                source=row["source_name"],
                method=row["method"],
                top_k=row["top_k"],
                num_eval=metrics["num_eval"],
                citation=metrics["citation_support_rate"],
                label=metrics["answer_label_accuracy"],
                supported=metrics["answer_supported_rate"],
                unsupported=metrics["unsupported_claim_rate"],
                entity_n=metrics["entity_evaluable"],
                entity=metrics["answer_evidence_entity_consistency"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]

    qrels_path = args.qrels or paths["pubmedqa_qrels"]
    question_entities_path = args.question_entities or "data/processed/pubmedqa_question_entities.jsonl"
    passage_entities_path = args.passage_entities or "data/processed/pubmedqa_passage_entities.jsonl"

    qrels_by_qid = qrels_map(read_jsonl(qrels_path))
    question_entities = entity_map(read_jsonl(question_entities_path), "question_id")
    passage_entities = entity_map(read_jsonl(passage_entities_path), "passage_id")

    details: list[dict[str, Any]] = []
    grouped_details: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    source_configs = []

    for source_arg in args.source:
        source_name, qa_path, evidence_path = parse_source(source_arg)
        qa_rows = read_jsonl(qa_path)
        evidence_by_qid = group_predictions(read_jsonl(evidence_path))
        source_configs.append({"source_name": source_name, "qa_predictions": qa_path, "evidence_predictions": evidence_path})

        for row in qa_rows:
            qid = str(row["question_id"])
            method = str(row["method"])
            top_k = int(row["top_k"])
            retrieved = evidence_by_qid.get(qid, [])[:top_k]
            retrieved_ids = {str(item["passage_id"]) for item in retrieved}
            gold_ids = qrels_by_qid.get(qid, set())
            question_entity_ids = question_entities.get(qid, set())
            retrieved_entity_ids: set[str] = set()
            for passage_id in retrieved_ids:
                retrieved_entity_ids.update(passage_entities.get(passage_id, set()))

            citation_supported = bool(gold_ids & retrieved_ids)
            label_supported = bool(row.get("correct", False))
            answer_supported = citation_supported and label_supported
            entity_evaluable = bool(question_entity_ids)
            entity_consistent = bool(entity_evaluable and (question_entity_ids & retrieved_entity_ids))

            detail = {
                "source_name": source_name,
                "question_id": qid,
                "method": method,
                "top_k": top_k,
                "gold_label": row.get("gold_label"),
                "predicted_label": row.get("predicted_label"),
                "citation_supported": citation_supported,
                "label_supported": label_supported,
                "answer_supported": answer_supported,
                "unsupported_claim": not answer_supported,
                "entity_evaluable": entity_evaluable,
                "entity_consistent": entity_consistent,
                "num_question_entities": len(question_entity_ids),
                "num_retrieved_entities": len(retrieved_entity_ids),
                "num_shared_entities": len(question_entity_ids & retrieved_entity_ids),
            }
            details.append(detail)
            grouped_details[(source_name, method, top_k)].append(detail)

    results = []
    for (source_name, method, top_k), rows in sorted(grouped_details.items()):
        results.append(
            {
                "source_name": source_name,
                "method": method,
                "top_k": top_k,
                "metrics": summarize_group(rows),
            }
        )

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "qrels": qrels_path,
            "question_entities": question_entities_path,
            "passage_entities": passage_entities_path,
            "sources": source_configs,
        },
        "metric_definitions": {
            "citation_support_rate": "Fraction of answer rows whose top-k retrieved passages include at least one gold PubMedQA evidence passage.",
            "answer_label_accuracy": "Fraction of answer rows where the selected yes/no/maybe label equals the PubMedQA gold label.",
            "answer_supported_rate": "Fraction of answer rows with both a supported citation set and the correct PubMedQA label.",
            "unsupported_claim_rate": "One minus answer_supported_rate; strict rule-based diagnostic, not an LLM judge.",
            "answer_evidence_entity_consistency": "Among questions with extracted entities, fraction whose top-k evidence shares at least one extracted entity with the question.",
        },
        "results": results,
        "notes": [
            "These diagnostics evaluate answer selection over PubMedQA labels, not free-form clinical generation.",
            "No LLM-as-judge is used; support is measured by qrels, gold labels, and deterministic entity overlap.",
        ],
    }
    write_json(args.output, payload)
    write_jsonl(args.details_output, details)
    Path(args.table_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.table_output).write_text(markdown_table(results), encoding="utf-8")
    print(
        {
            "metrics_output": args.output,
            "details_output": args.details_output,
            "table_output": args.table_output,
            "num_results": len(results),
            "num_details": len(details),
        }
    )


if __name__ == "__main__":
    main()
