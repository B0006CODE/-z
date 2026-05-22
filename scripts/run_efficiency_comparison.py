from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_cross_encoder_rerank import score_pairs_with_transformers
from run_kch_medrank import (
    build_all_feature_rows,
    feature_names_for,
    filter_qrels,
    make_ranker,
    matrix_for_qids,
    read_score_predictions,
    rerank_with_model,
    split_qids,
)
from src.evaluation.retrieval_metrics import evaluate_retrieval, group_qrels
from src.knowledge.mesh_hierarchy import load_mesh_hierarchy
from src.rerank.hypergraph import entity_map, mesh_map, relations_map
from src.retrieval.dense import passage_text
from src.utils import load_config, read_jsonl, resolve_torch_device, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MedCPT Cross-Encoder and KCH-MedRank reranking-stage inference cost."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--candidate-predictions", default="outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl")
    parser.add_argument("--semantic-predictions", default="outputs/retrieval/medcpt_dense_full_top100.jsonl")
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--mesh-hierarchy", default="data/external_knowledge/mesh_hierarchy_2026.jsonl")
    parser.add_argument("--relations", default=None)
    parser.add_argument("--kch-metrics", default="results/metrics/kch_medrank_enhanced_bioasq_metrics.json")
    parser.add_argument("--medcpt-metrics", default="results/metrics/medcpt_cross_encoder_enhanced_bioasq_test_top100_metrics.json")
    parser.add_argument("--output-json", default="results/metrics/efficiency_comparison_bioasq.json")
    parser.add_argument("--output-md", default="results/tables/efficiency_comparison_bioasq.md")
    parser.add_argument("--output-tex", default="paper/tables/reranking_efficiency.tex")
    parser.add_argument("--cross-encoder-model", default="ncbi/MedCPT-Cross-Encoder")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-m", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--split-modulo", type=int, default=5)
    parser.add_argument("--validation-remainders", type=int, nargs="+", default=[3])
    parser.add_argument("--test-remainders", type=int, nargs="+", default=[4])
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-cross-encoder", action="store_true")
    return parser.parse_args()


def now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def elapsed(start: float) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def group_predictions(rows: list[dict[str, Any]], top_m: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    for items in grouped.values():
        items.sort(key=lambda row: int(row["rank"]))
        del items[top_m:]
    return dict(grouped)


def flatten_by_qids(grouped: dict[str, list[dict[str, Any]]], qids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid in sorted(qids):
        rows.extend(grouped.get(qid, []))
    return rows


def load_metrics(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if value is not None else None


def build_cross_encoder_pairs(
    qids: set[str],
    candidates_by_qid: dict[str, list[dict[str, Any]]],
    question_by_id: dict[str, dict[str, Any]],
    passage_by_id: dict[str, dict[str, Any]],
) -> list[list[str]]:
    pairs: list[list[str]] = []
    for qid in sorted(qids):
        question = str(question_by_id[qid]["question"])
        for row in candidates_by_qid.get(qid, []):
            passage = passage_by_id.get(str(row["passage_id"]))
            if passage is None:
                continue
            pairs.append([question, passage_text(passage)])
    return pairs


def format_seconds(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def format_ms(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 1000.0:.2f}"


def format_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def write_markdown(path: str | Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "method",
        "device",
        "questions",
        "candidates",
        "rerank_seconds",
        "ms_per_query",
        "candidates_per_second",
        "recall@10",
        "mrr@10",
        "ndcg@10",
        "notes",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: str | Path, rows: list[dict[str, str]]) -> None:
    method_rows = []
    for row in rows:
        method_rows.append(
            " & ".join(
                [
                    row["method"],
                    row["rerank_seconds"],
                    row["ms_per_query"],
                    row["candidates_per_second"],
                    row["recall@10"],
                    row["mrr@10"],
                    row["ndcg@10"],
                    row["notes"],
                ]
            )
            + r" \\"
        )
    content = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Reranking-stage efficiency on the held-out BioASQ test split using the same enhanced top-100 candidate pool. Timing excludes first-stage candidate generation and offline LambdaMART training; KCH-MedRank includes test-time feature construction and LightGBM scoring, while MedCPT Cross-Encoder includes tokenization and forward scoring.}",
            r"\label{tab:reranking-efficiency}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lrrrrrrl}",
            r"\toprule",
            r"Method & Seconds & ms/query & Cand./s & Recall@10 & MRR@10 & nDCG@10 & Notes \\",
            r"\midrule",
            *method_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(seed)
    paths = config["paths"]
    device = resolve_torch_device(args.device)

    questions_path = args.questions or paths["questions"]
    corpus_path = args.corpus or paths["corpus"]
    qrels_path = args.qrels or paths.get("qrels", "data/processed/bioasq_qrels.jsonl")
    question_entities_path = args.question_entities or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    passage_entities_path = args.passage_entities or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")
    question_mesh_path = args.question_mesh or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")
    passage_mesh_path = args.passage_mesh or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")
    relations_path = args.relations or paths.get("primekg_relations", "data/external_knowledge/primekg_project_relations.jsonl")

    questions = read_jsonl(questions_path)
    corpus = read_jsonl(corpus_path)
    qrels = read_jsonl(qrels_path)
    candidates = read_jsonl(args.candidate_predictions)
    candidates_by_qid = group_predictions(candidates, args.top_m)
    all_qids = sorted(candidates_by_qid)
    splits = split_qids(
        all_qids,
        args.split_modulo,
        set(args.validation_remainders),
        set(args.test_remainders),
    )
    test_qids = set(sorted(splits["test"])[: args.sample_limit]) if args.sample_limit else set(splits["test"])
    train_validation_qids = splits["train"] | splits["validation"]
    question_by_id = {str(row["question_id"]): row for row in questions}
    passage_by_id = {str(row["passage_id"]): row for row in corpus}
    test_candidates = flatten_by_qids(candidates_by_qid, test_qids)
    train_validation_candidates = flatten_by_qids(candidates_by_qid, train_validation_qids)
    test_qrels = filter_qrels(qrels, test_qids)
    qrels_by_qid = group_qrels(qrels)

    medcpt_metrics = load_metrics(args.medcpt_metrics)
    kch_metrics_payload = load_metrics(args.kch_metrics)
    kch_metrics = kch_metrics_payload.get("setting_metrics", {}).get("Full KCH-MedRank") or load_metrics(
        "results/metrics/kch_medrank_enhanced_bioasq_full_kch_medrank_metrics.json"
    )
    selected = kch_metrics_payload["diagnostics"]["full_kch_medrank"]["selected"]

    cross_encoder_result: dict[str, Any] | None = None
    if not args.skip_cross_encoder:
        pairs = build_cross_encoder_pairs(test_qids, candidates_by_qid, question_by_id, passage_by_id)
        load_start = now()
        tokenizer = AutoTokenizer.from_pretrained(args.cross_encoder_model)
        model = AutoModelForSequenceClassification.from_pretrained(args.cross_encoder_model).to(device)
        model.eval()
        model_load_seconds = elapsed(load_start)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        score_start = now()
        _scores = score_pairs_with_transformers(
            model,
            tokenizer,
            pairs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )
        cross_encoder_seconds = elapsed(score_start)
        cross_encoder_result = {
            "method": "MedCPT Cross-Encoder",
            "model_name": args.cross_encoder_model,
            "backend": "transformers_sequence_classification",
            "device": device,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "model_load_seconds": model_load_seconds,
            "rerank_seconds": cross_encoder_seconds,
            "num_questions": len(test_qids),
            "num_candidates": len(pairs),
            "candidates_per_second": len(pairs) / cross_encoder_seconds if cross_encoder_seconds else None,
            "seconds_per_query": cross_encoder_seconds / len(test_qids) if test_qids else None,
            "gpu_peak_memory_mb": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None,
            "metrics": {
                "recall@10": metric_value(medcpt_metrics, "recall@10"),
                "mrr@10": metric_value(medcpt_metrics, "mrr@10"),
                "ndcg@10": metric_value(medcpt_metrics, "ndcg@10"),
            },
        }

    question_entities = entity_map(read_jsonl(question_entities_path), "question_id")
    passage_entities = entity_map(read_jsonl(passage_entities_path), "passage_id")
    question_mesh = mesh_map(read_jsonl(question_mesh_path), "question_id") if Path(question_mesh_path).exists() else {}
    passage_mesh = mesh_map(read_jsonl(passage_mesh_path), "passage_id") if Path(passage_mesh_path).exists() else {}
    mesh_hierarchy = load_mesh_hierarchy(read_jsonl(args.mesh_hierarchy)) if Path(args.mesh_hierarchy).exists() else {}
    entity_relations = relations_map(read_jsonl(relations_path)) if Path(relations_path).exists() else {}
    semantic_scores = read_score_predictions(args.semantic_predictions)
    feature_names = feature_names_for("full_kch_medrank")

    train_feature_start = now()
    train_validation_features = build_all_feature_rows(
        train_validation_candidates,
        question_entities,
        passage_entities,
        question_mesh,
        passage_mesh,
        mesh_hierarchy,
        entity_relations,
        semantic_scores,
        enable_mesh_hierarchy_graph_edges=False,
        structure="knowledge_hypergraph",
        top_k=args.top_k,
        rrf_k=60,
        iterations=3,
        damping=0.85,
        max_passage_entities=48,
        max_passage_mesh=32,
    )
    train_feature_seconds = elapsed(train_feature_start)
    train_x, train_y, train_group, _ = matrix_for_qids(
        train_validation_features,
        train_validation_qids,
        qrels_by_qid,
        feature_names,
    )
    train_start = now()
    ranker = make_ranker(
        seed,
        num_leaves=int(selected["num_leaves"]),
        learning_rate=float(selected["learning_rate"]),
        n_estimators=int(selected["n_estimators"]),
    )
    ranker.fit(train_x, train_y, group=train_group)
    train_seconds = elapsed(train_start)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    feature_start = now()
    test_features = build_all_feature_rows(
        test_candidates,
        question_entities,
        passage_entities,
        question_mesh,
        passage_mesh,
        mesh_hierarchy,
        entity_relations,
        semantic_scores,
        enable_mesh_hierarchy_graph_edges=False,
        structure="knowledge_hypergraph",
        top_k=args.top_k,
        rrf_k=60,
        iterations=3,
        damping=0.85,
        max_passage_entities=48,
        max_passage_mesh=32,
    )
    kch_feature_seconds = elapsed(feature_start)
    scoring_start = now()
    kch_predictions = rerank_with_model(
        ranker,
        test_features,
        test_qids,
        qrels_by_qid,
        feature_names,
        blend_weight=float(selected["blend_weight"]),
        top_k=args.top_k,
        retriever_name="full_kch_medrank_timing",
    )
    kch_scoring_seconds = elapsed(scoring_start)
    kch_total_seconds = kch_feature_seconds + kch_scoring_seconds
    measured_kch_metrics = evaluate_retrieval(test_qrels, kch_predictions, [10])

    kch_result = {
        "method": "KCH-MedRank",
        "device": "CPU tabular scoring",
        "selected_hyperparameters": selected,
        "offline_train_feature_seconds": train_feature_seconds,
        "offline_train_seconds": train_seconds,
        "rerank_feature_seconds": kch_feature_seconds,
        "rerank_scoring_seconds": kch_scoring_seconds,
        "rerank_seconds": kch_total_seconds,
        "num_questions": len(test_qids),
        "num_candidates": len(test_candidates),
        "candidates_per_second": len(test_candidates) / kch_total_seconds if kch_total_seconds else None,
        "seconds_per_query": kch_total_seconds / len(test_qids) if test_qids else None,
        "num_features": len(feature_names),
        "metrics": {
            "recall@10": metric_value(kch_metrics, "recall@10") if args.sample_limit is None else metric_value(measured_kch_metrics, "recall@10"),
            "mrr@10": metric_value(kch_metrics, "mrr@10") if args.sample_limit is None else metric_value(measured_kch_metrics, "mrr@10"),
            "ndcg@10": metric_value(kch_metrics, "ndcg@10") if args.sample_limit is None else metric_value(measured_kch_metrics, "ndcg@10"),
        },
    }

    speedup = None
    if cross_encoder_result and cross_encoder_result["rerank_seconds"] and kch_result["rerank_seconds"]:
        speedup = cross_encoder_result["rerank_seconds"] / kch_result["rerank_seconds"]

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "candidate_predictions": args.candidate_predictions,
        "semantic_predictions": args.semantic_predictions,
        "qrels": qrels_path,
        "top_m": args.top_m,
        "top_k": args.top_k,
        "sample_limit": args.sample_limit,
        "split": {
            "modulo": args.split_modulo,
            "validation_remainders": args.validation_remainders,
            "test_remainders": args.test_remainders,
            "test_qids": len(test_qids),
        },
        "fairness_note": (
            "Timing excludes first-stage candidate generation, one-time data/model loading, and offline LambdaMART training. "
            "MedCPT Cross-Encoder timing includes tokenizer preprocessing and neural forward scoring. "
            "KCH-MedRank timing includes test-time local feature construction, matrix creation, LightGBM scoring, and sorting."
        ),
        "cross_encoder": cross_encoder_result,
        "kch_medrank": kch_result,
        "speedup_cross_encoder_over_kch": speedup,
    }
    write_json(args.output_json, payload)

    rows = []
    if cross_encoder_result:
        rows.append(
            {
                "method": "MedCPT Cross-Encoder",
                "device": cross_encoder_result["device"],
                "questions": str(cross_encoder_result["num_questions"]),
                "candidates": str(cross_encoder_result["num_candidates"]),
                "rerank_seconds": format_seconds(cross_encoder_result["rerank_seconds"]),
                "ms_per_query": format_ms(cross_encoder_result["seconds_per_query"]),
                "candidates_per_second": format_seconds(cross_encoder_result["candidates_per_second"]),
                "recall@10": format_metric(cross_encoder_result["metrics"]["recall@10"]),
                "mrr@10": format_metric(cross_encoder_result["metrics"]["mrr@10"]),
                "ndcg@10": format_metric(cross_encoder_result["metrics"]["ndcg@10"]),
                "notes": "tokenization + forward",
            }
        )
    rows.append(
        {
            "method": "KCH-MedRank",
            "device": "CPU",
            "questions": str(kch_result["num_questions"]),
            "candidates": str(kch_result["num_candidates"]),
            "rerank_seconds": format_seconds(kch_result["rerank_seconds"]),
            "ms_per_query": format_ms(kch_result["seconds_per_query"]),
            "candidates_per_second": format_seconds(kch_result["candidates_per_second"]),
            "recall@10": format_metric(kch_result["metrics"]["recall@10"]),
            "mrr@10": format_metric(kch_result["metrics"]["mrr@10"]),
            "ndcg@10": format_metric(kch_result["metrics"]["ndcg@10"]),
            "notes": "features + LightGBM",
        }
    )
    write_markdown(args.output_md, rows)
    write_latex(args.output_tex, rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
