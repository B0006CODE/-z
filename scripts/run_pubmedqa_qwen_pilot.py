from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import group_qrels
from src.utils import read_jsonl, write_json


LABELS = ["yes", "no", "maybe"]
STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "answer",
    "because",
    "before",
    "being",
    "between",
    "could",
    "does",
    "evidence",
    "from",
    "have",
    "into",
    "only",
    "other",
    "results",
    "should",
    "show",
    "shown",
    "study",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "with",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "would",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled DashScope Qwen PubMedQA generation pilot."
    )
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--questions", default="data/processed/pubmedqa_pqa_labeled_questions.jsonl")
    parser.add_argument("--corpus", default="data/processed/pubmedqa_pqa_labeled_corpus.jsonl")
    parser.add_argument("--qrels", default="data/processed/pubmedqa_pqa_labeled_qrels.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default="outputs/generation/pubmedqa_qwen3_8b_pilot.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/pubmedqa_qwen3_8b_pilot_metrics.json")
    parser.add_argument("--table-output", default="results/tables/pubmedqa_qwen3_8b_pilot.md")
    parser.add_argument(
        "--method",
        action="append",
        nargs=3,
        metavar=("NAME", "PREDICTIONS", "DISPLAY"),
        required=True,
        help="Method name, prediction JSONL path, and display label.",
    )
    return parser.parse_args()


def load_env(path: str | Path) -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = Path(path)
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def group_predictions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    for qid_rows in grouped.values():
        qid_rows.sort(key=lambda row: int(row["rank"]))
    return dict(grouped)


def load_questions(path: str | Path) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        label = str(row.get("metadata", {}).get("final_decision", "")).strip().lower()
        if label in LABELS:
            questions[str(row["question_id"])] = row
    return questions


def load_corpus(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row["passage_id"]): row for row in read_jsonl(path)}


def compact_text(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars].rstrip()


def build_prompt(question: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    evidence_lines = []
    for idx, row in enumerate(evidence, start=1):
        title = compact_text(str(row.get("title", "")), 140)
        text = compact_text(str(row.get("text", "")), 900)
        evidence_lines.append(f"[P{idx}] {title}. {text}")
    return (
        "Answer the PubMedQA question using only the cited evidence passages.\n"
        "Choose exactly one label from: yes, no, maybe.\n"
        "Return strict JSON with keys: answer, citations, rationale.\n"
        "The citations value must be a list of passage ids such as [\"P1\", \"P2\"].\n\n"
        f"Question: {question['question']}\n\n"
        "Evidence:\n"
        + "\n".join(evidence_lines)
    )


def call_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a cautious biomedical evidence-grounded QA evaluator. "
                    "Use only the provided evidence and return strict JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"])


def parse_response(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    parsed: dict[str, Any] = {}
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                parsed = value
        except json.JSONDecodeError:
            parsed = {}
    answer = str(parsed.get("answer", "")).strip().lower()
    if answer not in LABELS:
        label_match = re.search(r"\b(yes|no|maybe)\b", cleaned.lower())
        answer = label_match.group(1) if label_match else "maybe"
    citations_raw = parsed.get("citations", [])
    if isinstance(citations_raw, str):
        citations = re.findall(r"P\d+", citations_raw)
    elif isinstance(citations_raw, list):
        citations = [str(item).strip() for item in citations_raw if re.fullmatch(r"P\d+", str(item).strip())]
    else:
        citations = []
    return {
        "answer": answer,
        "citations": citations,
        "rationale": str(parsed.get("rationale", "")).strip(),
        "raw_response": text,
    }


def content_terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text.lower()):
        if token not in STOPWORDS and not token.isdigit():
            terms.add(token)
    return terms


def entity_consistency(rationale: str, evidence_text: str) -> float | None:
    rationale_terms = content_terms(rationale)
    if not rationale_terms:
        return None
    evidence_terms = content_terms(evidence_text)
    if not evidence_terms:
        return None
    return len(rationale_terms & evidence_terms) / len(rationale_terms)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [row["gold_answer"] for row in records]
    pred = [row["predicted_answer"] for row in records]
    citation_support = [1.0 if row["citation_supported"] else 0.0 for row in records]
    unsupported = [1.0 if row["unsupported_claim"] else 0.0 for row in records]
    consistency_values = [
        float(row["answer_evidence_entity_consistency"])
        for row in records
        if row["answer_evidence_entity_consistency"] is not None
    ]
    return {
        "num_eval": len(records),
        "accuracy": float(accuracy_score(gold, pred)) if records else 0.0,
        "macro_f1": float(f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)) if records else 0.0,
        "citation_support_rate": float(sum(citation_support) / len(citation_support)) if records else 0.0,
        "unsupported_claim_rate": float(sum(unsupported) / len(unsupported)) if records else 0.0,
        "entity_consistency_evaluable": len(consistency_values),
        "answer_evidence_entity_consistency": (
            float(sum(consistency_values) / len(consistency_values)) if consistency_values else 0.0
        ),
    }


def markdown_table(metrics: dict[str, Any]) -> str:
    lines = [
        "| method | num_eval | accuracy | macro_f1 | citation_support | unsupported_claim | entity_evaluable | entity_consistency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in metrics["method_order"]:
        row = metrics["methods"][method]
        lines.append(
            "| {display} | {num_eval} | {accuracy:.4f} | {macro_f1:.4f} | {citation:.4f} | {unsupported:.4f} | {entity_n} | {entity:.4f} |".format(
                display=row["display_name"],
                num_eval=row["num_eval"],
                accuracy=row["accuracy"],
                macro_f1=row["macro_f1"],
                citation=row["citation_support_rate"],
                unsupported=row["unsupported_claim_rate"],
                entity_n=row["entity_consistency_evaluable"],
                entity=row["answer_evidence_entity_consistency"],
            )
        )
    lines.append("")
    lines.append(
        "Pilot diagnostic: fixed prompt, top-k evidence, deterministic decoding, and no model fine-tuning. "
        "Citation support requires at least one model-cited passage to overlap a PubMedQA gold evidence passage; "
        "unsupported claim rate is the complement of this citation-support test."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    env = load_env(args.env_path)
    model = env.get("DASHSCOPE_MODEL")
    base_url = env.get("DASHSCOPE_BASE_URL")
    api_key = env.get("DASHSCOPE_API_KEY")
    if not model or not base_url or not api_key:
        raise RuntimeError("DASHSCOPE_MODEL, DASHSCOPE_BASE_URL, and DASHSCOPE_API_KEY must be set in .env.")

    questions = load_questions(args.questions)
    corpus = load_corpus(args.corpus)
    qrels = group_qrels(read_jsonl(args.qrels))
    methods = [
        {"name": name, "path": path, "display": display, "predictions": group_predictions(read_jsonl(path))}
        for name, path, display in args.method
    ]
    common_qids = sorted(set(questions) & set(qrels).intersection(*(set(m["predictions"]) for m in methods)))
    selected_qids = common_qids[: args.max_questions]
    if not selected_qids:
        raise RuntimeError("No common PubMedQA qids found across questions, qrels, and method predictions.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys: set[tuple[str, str]] = set()
    if output_path.exists():
        for row in read_jsonl(output_path):
            existing_keys.add((str(row["method"]), str(row["question_id"])))

    records: list[dict[str, Any]] = []
    with output_path.open("a", encoding="utf-8") as f:
        for method in methods:
            for qid in selected_qids:
                key = (method["name"], qid)
                if key in existing_keys:
                    continue
                prediction_rows = method["predictions"][qid][: args.top_k]
                evidence = []
                passage_id_by_label: dict[str, str] = {}
                evidence_text_parts: list[str] = []
                for idx, pred_row in enumerate(prediction_rows, start=1):
                    passage_id = str(pred_row["passage_id"])
                    passage = corpus.get(passage_id, {"passage_id": passage_id, "title": "", "text": ""})
                    evidence.append(passage)
                    passage_id_by_label[f"P{idx}"] = passage_id
                    evidence_text_parts.append(f"{passage.get('title', '')} {passage.get('text', '')}")
                prompt = build_prompt(questions[qid], evidence)
                raw_response = call_chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                parsed = parse_response(raw_response)
                cited_passage_ids = [passage_id_by_label[c] for c in parsed["citations"] if c in passage_id_by_label]
                gold_ids = set(qrels[qid])
                citation_supported = bool(gold_ids & set(cited_passage_ids))
                consistency = entity_consistency(parsed["rationale"], " ".join(evidence_text_parts))
                record = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "method": method["name"],
                    "display_name": method["display"],
                    "question_id": qid,
                    "gold_answer": questions[qid]["metadata"]["final_decision"],
                    "predicted_answer": parsed["answer"],
                    "citations": parsed["citations"],
                    "cited_passage_ids": cited_passage_ids,
                    "citation_supported": citation_supported,
                    "unsupported_claim": not citation_supported,
                    "answer_evidence_entity_consistency": consistency,
                    "top_k": args.top_k,
                    "model": model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "prompt_version": "pubmedqa_json_cited_label_v1",
                    "raw_response": parsed["raw_response"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                records.append(record)
                if args.sleep > 0:
                    time.sleep(args.sleep)

    all_rows = [
        row
        for row in read_jsonl(output_path)
        if str(row.get("question_id")) in set(selected_qids)
        and str(row.get("method")) in {method["name"] for method in methods}
    ]
    metrics = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": "qiaojin/PubMedQA/pqa_labeled",
        "model": model,
        "base_url": base_url,
        "top_k": args.top_k,
        "max_questions": args.max_questions,
        "num_common_qids": len(common_qids),
        "num_selected_qids": len(selected_qids),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "prompt_version": "pubmedqa_json_cited_label_v1",
        "metric_definitions": {
            "citation_support_rate": "Fraction of responses with at least one model-cited passage that overlaps a PubMedQA gold evidence passage.",
            "unsupported_claim_rate": "One minus citation_support_rate under this rule-based pilot protocol.",
            "answer_evidence_entity_consistency": "Mean fraction of non-stopword rationale terms that appear in the supplied evidence, over evaluable responses.",
        },
        "method_order": [method["name"] for method in methods],
        "methods": {},
    }
    for method in methods:
        method_rows = [row for row in all_rows if row["method"] == method["name"]]
        summary = summarize(method_rows)
        summary["display_name"] = method["display"]
        summary["predictions"] = method["path"]
        metrics["methods"][method["name"]] = summary
    write_json(args.metrics_output, metrics)
    Path(args.table_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.table_output).write_text(markdown_table(metrics), encoding="utf-8")
    print(
        {
            "output": args.output,
            "metrics_output": args.metrics_output,
            "table_output": args.table_output,
            "model": model,
            "num_selected_qids": len(selected_qids),
            "new_records": len(records),
        }
    )


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DashScope request failed with HTTP {exc.code}: {message[:500]}") from exc
