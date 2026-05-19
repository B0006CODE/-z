from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.pubmedqa_answering import (
    PUBMEDQA_LABELS,
    build_answer_text,
    evaluate_labels,
    evidence_coverage_at_k,
    group_predictions,
    lexical_rule_predict,
    majority_label,
    split_qids,
    train_tfidf_logreg,
)
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lightweight PubMedQA yes/no/maybe answer selection.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--answer-labels", default=None)
    parser.add_argument("--predictions", required=True, help="Evaluation evidence predictions JSONL.")
    parser.add_argument(
        "--train-predictions",
        default=None,
        help="Training evidence predictions JSONL. Defaults to --predictions when it contains train qids.",
    )
    parser.add_argument("--output", default="outputs/generation/pubmedqa_qa_predictions.jsonl")
    parser.add_argument("--metrics-output", default="results/metrics/pubmedqa_qa_metrics.json")
    parser.add_argument("--table-output", default="results/tables/pubmedqa_qa_accuracy.md")
    parser.add_argument("--methods", nargs="+", default=["majority", "lexical_rule", "tfidf_logreg"])
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--eval-split", choices=["validation", "test", "all"], default="test")
    parser.add_argument("--split-modulo", type=int, default=5)
    parser.add_argument("--validation-remainders", type=int, nargs="+", default=[3])
    parser.add_argument("--test-remainders", type=int, nargs="+", default=[4])
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit eval qids for sanity checks.")
    parser.add_argument("--include-evidence", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def label_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    labels = {}
    for row in rows:
        label = str(row.get("final_decision", "")).strip().lower()
        if label in PUBMEDQA_LABELS:
            labels[str(row["question_id"])] = label
    return labels


def qrels_map(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[str(row["question_id"])].add(str(row["passage_id"]))
    return dict(grouped)


def make_examples(
    qids: list[str],
    *,
    labels_by_qid: dict[str, str],
    question_by_qid: dict[str, str],
    corpus_by_pid: dict[str, dict[str, Any]],
    predictions_by_qid: dict[str, list[dict[str, Any]]],
    top_k: int,
    include_evidence: bool,
) -> tuple[list[str], list[str], list[str]]:
    used_qids = [qid for qid in qids if qid in labels_by_qid]
    texts = [
        build_answer_text(
            qid,
            question_by_qid=question_by_qid,
            corpus_by_pid=corpus_by_pid,
            predictions_by_qid=predictions_by_qid,
            top_k=top_k,
            include_evidence=include_evidence,
        )
        for qid in used_qids
    ]
    labels = [labels_by_qid[qid] for qid in used_qids]
    return used_qids, texts, labels


def markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| predictions | method | top_k | eval_split | num_eval | evidence_hit@k | accuracy | macro_f1 | yes_f1 | no_f1 | maybe_f1 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        per_label = row["metrics"]["per_label"]
        lines.append(
            "| {predictions} | {method} | {top_k} | {eval_split} | {num_eval} | {coverage:.4f} | {accuracy:.4f} | {macro_f1:.4f} | {yes_f1:.4f} | {no_f1:.4f} | {maybe_f1:.4f} |".format(
                predictions=row["prediction_name"],
                method=row["method"],
                top_k=row["top_k"],
                eval_split=row["eval_split"],
                num_eval=row["num_eval"],
                coverage=row["evidence_hit_at_k"],
                accuracy=row["metrics"]["accuracy"],
                macro_f1=row["metrics"]["macro_f1"],
                yes_f1=per_label["yes"]["f1"],
                no_f1=per_label["no"]["f1"],
                maybe_f1=per_label["maybe"]["f1"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    paths = config["paths"]

    questions_path = args.questions or paths["pubmedqa_questions"]
    corpus_path = args.corpus or paths["pubmedqa_corpus"]
    qrels_path = args.qrels or paths["pubmedqa_qrels"]
    labels_path = args.answer_labels or paths["pubmedqa_answer_labels"]

    questions = read_jsonl(questions_path)
    corpus = read_jsonl(corpus_path)
    qrels = read_jsonl(qrels_path)
    answer_labels = read_jsonl(labels_path)
    eval_predictions = group_predictions(read_jsonl(args.predictions))
    train_predictions = group_predictions(read_jsonl(args.train_predictions or args.predictions))

    question_by_qid = {str(row["question_id"]): str(row.get("question", "")) for row in questions}
    corpus_by_pid = {str(row["passage_id"]): row for row in corpus}
    labels_by_qid = label_map(answer_labels)
    qrels_by_qid = qrels_map(qrels)

    splits = split_qids(
        labels_by_qid.keys(),
        modulo=args.split_modulo,
        validation_remainders=set(args.validation_remainders),
        test_remainders=set(args.test_remainders),
    )
    train_qids = sorted(splits["train"])
    eval_qids = sorted(qid for qid in splits[args.eval_split] if qid in eval_predictions)
    if args.sample_limit is not None:
        eval_qids = eval_qids[: args.sample_limit]
    if not eval_qids:
        raise ValueError("No evaluation qids remain after applying split, prediction coverage, and sample limit.")

    prediction_name = Path(args.predictions).stem
    all_results: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []

    for top_k in args.top_ks:
        train_used_qids, train_texts, train_y = make_examples(
            train_qids,
            labels_by_qid=labels_by_qid,
            question_by_qid=question_by_qid,
            corpus_by_pid=corpus_by_pid,
            predictions_by_qid=train_predictions,
            top_k=top_k,
            include_evidence=args.include_evidence,
        )
        eval_used_qids, eval_texts, eval_y = make_examples(
            eval_qids,
            labels_by_qid=labels_by_qid,
            question_by_qid=question_by_qid,
            corpus_by_pid=corpus_by_pid,
            predictions_by_qid=eval_predictions,
            top_k=top_k,
            include_evidence=args.include_evidence,
        )
        majority = majority_label(train_y)
        coverage = evidence_coverage_at_k(eval_used_qids, eval_predictions, qrels_by_qid, top_k=top_k)

        for method in args.methods:
            if method == "majority":
                pred_y = [majority for _ in eval_y]
                confidences = [None for _ in pred_y]
            elif method == "lexical_rule":
                pred_y = [lexical_rule_predict(text) for text in eval_texts]
                confidences = [None for _ in pred_y]
            elif method == "tfidf_logreg":
                if len(set(train_y)) < 2:
                    raise ValueError("TF-IDF logistic regression requires at least two labels in the training split.")
                model = train_tfidf_logreg(train_texts, train_y, seed=seed)
                pred_y = [str(label) for label in model.predict(eval_texts)]
                confidences = []
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(eval_texts)
                    classes = list(model.classes_)
                    for probs, label in zip(probabilities, pred_y, strict=False):
                        confidences.append(float(probs[classes.index(label)]))
                else:
                    confidences = [None for _ in pred_y]
            else:
                raise ValueError(f"Unsupported method: {method}")

            metrics = evaluate_labels(eval_y, pred_y)
            result = {
                "prediction_name": prediction_name,
                "prediction_path": args.predictions,
                "train_prediction_path": args.train_predictions or args.predictions,
                "method": method,
                "top_k": top_k,
                "eval_split": args.eval_split,
                "num_train": len(train_used_qids),
                "num_eval": len(eval_used_qids),
                "include_evidence": args.include_evidence,
                "train_label_counts": dict(sorted(Counter(train_y).items())),
                "eval_label_counts": dict(sorted(Counter(eval_y).items())),
                "evidence_hit_at_k": coverage,
                "metrics": metrics,
            }
            all_results.append(result)

            for qid, gold, pred, confidence in zip(eval_used_qids, eval_y, pred_y, confidences, strict=False):
                output_rows.append(
                    {
                        "question_id": qid,
                        "method": method,
                        "top_k": top_k,
                        "gold_label": gold,
                        "predicted_label": pred,
                        "confidence": confidence,
                        "correct": gold == pred,
                        "evidence_hit_at_k": bool(
                            qrels_by_qid.get(qid, set())
                            & {str(row["passage_id"]) for row in eval_predictions.get(qid, [])[:top_k]}
                        ),
                    }
                )

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "questions": questions_path,
            "corpus": corpus_path,
            "qrels": qrels_path,
            "answer_labels": labels_path,
            "predictions": args.predictions,
            "train_predictions": args.train_predictions or args.predictions,
            "methods": args.methods,
            "top_ks": args.top_ks,
            "eval_split": args.eval_split,
            "sample_limit": args.sample_limit,
            "include_evidence": args.include_evidence,
        },
        "results": all_results,
        "notes": [
            "This is answer selection over PubMedQA yes/no/maybe labels, not free-form clinical advice generation.",
            "TF-IDF logistic regression is trained only on the deterministic train split.",
            "For rerankers that only output held-out predictions, pass a full first-stage file via --train-predictions.",
        ],
    }
    write_json(args.metrics_output, payload)
    write_jsonl(args.output, output_rows)
    Path(args.table_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.table_output).write_text(markdown_table(all_results), encoding="utf-8")
    print(
        {
            "metrics_output": args.metrics_output,
            "predictions_output": args.output,
            "table_output": args.table_output,
            "num_results": len(all_results),
            "num_prediction_rows": len(output_rows),
        }
    )


if __name__ == "__main__":
    main()
