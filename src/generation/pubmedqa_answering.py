from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline


PUBMEDQA_LABELS = ["yes", "no", "maybe"]


def qid_bucket(qid: str, modulo: int) -> int:
    if qid.isdigit():
        return int(qid) % modulo
    return sum(ord(char) for char in qid) % modulo


def split_qids(
    qids: Iterable[str],
    *,
    modulo: int = 5,
    validation_remainders: set[int] | None = None,
    test_remainders: set[int] | None = None,
) -> dict[str, set[str]]:
    validation_remainders = validation_remainders or {3}
    test_remainders = test_remainders or {4}
    qid_set = set(qids)
    validation = {qid for qid in qid_set if qid_bucket(qid, modulo) in validation_remainders}
    test = {qid for qid in qid_set if qid_bucket(qid, modulo) in test_remainders}
    train = qid_set - validation - test
    return {"train": train, "validation": validation, "test": test, "all": qid_set}


def group_predictions(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    for qid in grouped:
        grouped[qid].sort(key=lambda row: int(row.get("rank", 10**9)))
    return dict(grouped)


def evidence_coverage_at_k(
    qids: Iterable[str],
    predictions_by_qid: dict[str, list[dict[str, Any]]],
    qrels_by_qid: dict[str, set[str]],
    *,
    top_k: int,
) -> float:
    total = 0
    covered = 0
    for qid in qids:
        gold = qrels_by_qid.get(qid, set())
        if not gold:
            continue
        total += 1
        retrieved = {str(row["passage_id"]) for row in predictions_by_qid.get(qid, [])[:top_k]}
        covered += int(bool(gold & retrieved))
    return covered / total if total else 0.0


def build_answer_text(
    qid: str,
    *,
    question_by_qid: dict[str, str],
    corpus_by_pid: dict[str, dict[str, Any]],
    predictions_by_qid: dict[str, list[dict[str, Any]]],
    top_k: int,
    include_evidence: bool = True,
) -> str:
    question = question_by_qid.get(qid, "")
    if not include_evidence:
        return f"Question: {question}"

    evidence_parts = []
    for row in predictions_by_qid.get(qid, [])[:top_k]:
        passage = corpus_by_pid.get(str(row["passage_id"]), {})
        title = str(passage.get("title", "")).strip()
        text = str(passage.get("text", "")).strip()
        if title or text:
            evidence_parts.append(f"{title}\n{text}".strip())
    evidence = "\n\n".join(evidence_parts)
    return f"Question: {question}\n\nEvidence:\n{evidence}".strip()


def lexical_rule_predict(text: str) -> str:
    lowered = f" {text.lower()} "
    no_patterns = [
        " no ",
        " not ",
        " without ",
        " failed to ",
        " failure to ",
        " did not ",
        " does not ",
        " do not ",
        " was not ",
        " were not ",
        " cannot ",
        " could not ",
        " lack of ",
        " lacks ",
        " unlikely ",
        " no significant ",
        " not associated ",
        " negative ",
    ]
    maybe_patterns = [
        " may ",
        " might ",
        " could ",
        " possible ",
        " possibly ",
        " suggest ",
        " suggests ",
        " uncertain ",
        " unclear ",
        " inconclusive ",
        " insufficient ",
        " limited evidence ",
        " further studies ",
    ]
    no_count = sum(lowered.count(pattern) for pattern in no_patterns)
    maybe_count = sum(lowered.count(pattern) for pattern in maybe_patterns)
    if no_count > 0 and no_count >= maybe_count:
        return "no"
    if maybe_count > 0:
        return "maybe"
    return "yes"


def train_tfidf_logreg(texts: list[str], labels: list[str], *, seed: int) -> Any:
    return make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        ),
        LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="liblinear",
        ),
    ).fit(texts, labels)


def majority_label(labels: Iterable[str]) -> str:
    counts = Counter(label for label in labels if label in PUBMEDQA_LABELS)
    if not counts:
        return "yes"
    return sorted(counts.items(), key=lambda item: (-item[1], PUBMEDQA_LABELS.index(item[0])))[0][0]


def evaluate_labels(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=PUBMEDQA_LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=PUBMEDQA_LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=PUBMEDQA_LABELS, average="macro", zero_division=0))
        if y_true
        else 0.0,
        "label_order": PUBMEDQA_LABELS,
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_label": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in PUBMEDQA_LABELS
        },
    }


def prediction_confidence(model: Any, texts: list[str], predictions: list[str]) -> list[float | None]:
    if not hasattr(model, "predict_proba"):
        return [None for _ in predictions]
    probabilities = model.predict_proba(texts)
    classes = list(model.classes_)
    confidences: list[float | None] = []
    for probs, label in zip(probabilities, predictions, strict=False):
        if label in classes:
            confidences.append(float(probs[classes.index(label)]))
        else:
            confidences.append(float(np.max(probs)))
    return confidences
