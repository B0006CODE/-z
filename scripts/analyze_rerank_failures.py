from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import group_predictions, group_qrels
from src.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze rescued and lost gold evidence after reranking.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def index_by_id(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows}


def rank_lookup(rows_by_qid: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    return {
        qid: {str(row["passage_id"]): int(row["rank"]) for row in rows}
        for qid, rows in rows_by_qid.items()
    }


def mesh_lookup(path: str | None, key: str) -> dict[str, set[str]]:
    if not path:
        return {}
    mapped: dict[str, set[str]] = {}
    for row in read_jsonl(path):
        mapped[str(row[key])] = {
            str(term.get("normalized") or term.get("mesh_name") or "").strip()
            for term in row.get("mesh_terms", [])
            if str(term.get("normalized") or term.get("mesh_name") or "").strip()
        }
    return mapped


def truncate(text: str, max_chars: int = 360) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def case_record(
    *,
    qid: str,
    pid: str,
    baseline_rank: int | None,
    candidate_rank: int | None,
    questions: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    question_mesh: dict[str, set[str]],
    passage_mesh: dict[str, set[str]],
) -> dict[str, Any]:
    q_mesh = question_mesh.get(qid, set())
    p_mesh = passage_mesh.get(pid, set())
    passage = corpus.get(pid, {})
    return {
        "question_id": qid,
        "passage_id": pid,
        "baseline_rank": baseline_rank,
        "candidate_rank": candidate_rank,
        "rank_delta": None if baseline_rank is None or candidate_rank is None else candidate_rank - baseline_rank,
        "question": questions.get(qid, {}).get("question", ""),
        "passage_title": passage.get("title", ""),
        "passage_text": passage.get("text", ""),
        "mesh_overlap": sorted(q_mesh & p_mesh)[:12],
        "question_mesh": sorted(q_mesh)[:12],
        "passage_mesh": sorted(p_mesh)[:12],
    }


def write_markdown(path: str | Path, payload: dict[str, Any]) -> None:
    top_k = payload["top_k"]
    baseline_label = payload["baseline_label"]
    candidate_label = payload["candidate_label"]
    summary = payload["summary"]
    lines = [
        "# Reranking Failure Analysis",
        "",
        f"Comparison: {baseline_label} -> {candidate_label}; top-k = {top_k}.",
        "",
        "| diagnostic | value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    for section_key, title in [("lost_cases", "Lost Gold Evidence"), ("rescued_cases", "Rescued Gold Evidence")]:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| question_id | passage_id | baseline rank | candidate rank | rank delta | MeSH overlap |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for case in payload[section_key]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        escape_md(case["question_id"]),
                        escape_md(case["passage_id"]),
                        str(case["baseline_rank"] or ">100"),
                        str(case["candidate_rank"] or ">100"),
                        str(case["rank_delta"] if case["rank_delta"] is not None else "n/a"),
                        escape_md(", ".join(case["mesh_overlap"]) or "none"),
                    ]
                )
                + " |"
            )
        lines.append("")
        for idx, case in enumerate(payload[section_key], start=1):
            lines.append(f"### {title} {idx}: {case['question_id']}")
            lines.append("")
            lines.append(f"Question: {escape_md(case['question'])}")
            lines.append("")
            title_text = f"{case['passage_title']} " if case["passage_title"] else ""
            lines.append(f"Gold passage: {escape_md(truncate(title_text + case['passage_text']))}")
            lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    questions = index_by_id(read_jsonl(args.questions), "question_id")
    corpus = index_by_id(read_jsonl(args.corpus), "passage_id")
    qrels = group_qrels(read_jsonl(args.qrels))
    baseline_ranks = rank_lookup(group_predictions(read_jsonl(args.baseline_predictions)))
    candidate_ranks = rank_lookup(group_predictions(read_jsonl(args.candidate_predictions)))
    question_mesh = mesh_lookup(args.question_mesh, "question_id")
    passage_mesh = mesh_lookup(args.passage_mesh, "passage_id")

    lost_cases: list[dict[str, Any]] = []
    rescued_cases: list[dict[str, Any]] = []
    qids = sorted(set(qrels) & set(baseline_ranks) & set(candidate_ranks))
    questions_with_lost = set()
    questions_with_rescued = set()
    for qid in qids:
        for pid in qrels[qid]:
            baseline_rank = baseline_ranks.get(qid, {}).get(pid)
            candidate_rank = candidate_ranks.get(qid, {}).get(pid)
            baseline_in_topk = baseline_rank is not None and baseline_rank <= args.top_k
            candidate_in_topk = candidate_rank is not None and candidate_rank <= args.top_k
            if baseline_in_topk and not candidate_in_topk:
                questions_with_lost.add(qid)
                lost_cases.append(
                    case_record(
                        qid=qid,
                        pid=pid,
                        baseline_rank=baseline_rank,
                        candidate_rank=candidate_rank,
                        questions=questions,
                        corpus=corpus,
                        question_mesh=question_mesh,
                        passage_mesh=passage_mesh,
                    )
                )
            if not baseline_in_topk and candidate_in_topk:
                questions_with_rescued.add(qid)
                rescued_cases.append(
                    case_record(
                        qid=qid,
                        pid=pid,
                        baseline_rank=baseline_rank,
                        candidate_rank=candidate_rank,
                        questions=questions,
                        corpus=corpus,
                        question_mesh=question_mesh,
                        passage_mesh=passage_mesh,
                    )
                )

    lost_cases.sort(key=lambda row: (row["baseline_rank"] or 10**9, -(row["candidate_rank"] or 10**9), row["question_id"]))
    rescued_cases.sort(key=lambda row: (row["candidate_rank"] or 10**9, -(row["baseline_rank"] or 10**9), row["question_id"]))
    payload = {
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "top_k": args.top_k,
        "summary": {
            "paired_questions": len(qids),
            "questions_with_lost_gold": len(questions_with_lost),
            "lost_gold_passages": len(lost_cases),
            "questions_with_rescued_gold": len(questions_with_rescued),
            "rescued_gold_passages": len(rescued_cases),
        },
        "lost_cases": lost_cases[: args.max_cases],
        "rescued_cases": rescued_cases[: args.max_cases],
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print({"output_json": args.output_json, "output_md": args.output_md, "summary": payload["summary"]})


if __name__ == "__main__":
    main()
