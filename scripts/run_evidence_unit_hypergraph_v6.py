from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from run_evidence_unit_hypergraph_v5 import (  # noqa: E402
    FLAT_EVIDENCE_UNIT_FEATURES,
    HYPERGRAPH_EVIDENCE_UNIT_FEATURES,
    RETRIEVAL_FEATURES,
    feature_matrix,
    flatten_predictions,
    load_map,
    load_pubmed_metadata,
    merge_entity_maps,
    qid_sort_key,
    score_model,
    select_prediction_rows,
    split_qids,
    load_pubtator_entity_map,
)
from src.evaluation.retrieval_metrics import evaluate_retrieval, group_qrels  # noqa: E402
from src.hypergraph.evidence_unit import EvidenceUnitConfig  # noqa: E402
from src.hypergraph.evidence_unit_v6 import build_evidence_unit_v6_rows  # noqa: E402
from src.knowledge.mesh_hierarchy import load_mesh_hierarchy  # noqa: E402
from src.rerank.hypergraph import entity_map, mesh_map  # noqa: E402
from src.rerank.lambdamart import make_lambdamart_ranker  # noqa: E402
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl  # noqa: E402


def parse_int_grid(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def parse_float_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return values


V6_TWO_LAYER_FEATURES = [
    "evidence_hyperedge_quality",
    "topic_hyperedge_support",
    "evidence_topic_bridge_score",
    "topic_cluster_support",
    "evidence_cluster_support",
    "pico_completeness",
    "query_topic_coverage",
    "topic_token_count",
    "evidence_token_count",
]

V6_SUPPORT_FEATURES = [
    "support_sufficiency_gain",
    "support_necessity_gain",
    "minimal_support_score",
    "hard_rescue_score",
    "top10_evidence_quality_gap",
]

FEATURE_SETS = {
    "source_order": [],
    "retrieval_ltr": RETRIEVAL_FEATURES,
    "flat_evidence_unit_ltr": [*RETRIEVAL_FEATURES, *FLAT_EVIDENCE_UNIT_FEATURES],
    "v6_two_layer_hypergraph_ltr": [
        *RETRIEVAL_FEATURES,
        *FLAT_EVIDENCE_UNIT_FEATURES,
        *HYPERGRAPH_EVIDENCE_UNIT_FEATURES,
        *V6_TWO_LAYER_FEATURES,
        *V6_SUPPORT_FEATURES,
    ],
    "without_topic_hyperedges": [
        *RETRIEVAL_FEATURES,
        *FLAT_EVIDENCE_UNIT_FEATURES,
        *HYPERGRAPH_EVIDENCE_UNIT_FEATURES,
        *V6_SUPPORT_FEATURES,
    ],
    "without_sufficiency_necessity": [
        *RETRIEVAL_FEATURES,
        *FLAT_EVIDENCE_UNIT_FEATURES,
        *HYPERGRAPH_EVIDENCE_UNIT_FEATURES,
        *V6_TWO_LAYER_FEATURES,
    ],
    "without_hard_rescue": [
        *RETRIEVAL_FEATURES,
        *FLAT_EVIDENCE_UNIT_FEATURES,
        *HYPERGRAPH_EVIDENCE_UNIT_FEATURES,
        *[feature for feature in V6_TWO_LAYER_FEATURES],
        *[feature for feature in V6_SUPPORT_FEATURES if feature != "hard_rescue_score"],
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v6 Dou-style two-layer evidence/topic hypergraph reranking with hard-subset-aware LTR."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictions", default="outputs/retrieval/concept_hg_shared_clusters_sample100_top300.jsonl")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--mesh-hierarchy", default=None)
    parser.add_argument("--pubmed-metadata", default=None)
    parser.add_argument("--use-pubtator-concepts", action="store_true")
    parser.add_argument("--question-pubtator-concepts", default="data/processed/bioasq_question_pubtator_concepts.jsonl")
    parser.add_argument("--passage-pubtator-concepts", default="data/processed/bioasq_passage_pubtator_concepts.jsonl")
    parser.add_argument("--qid-file", default=None)
    parser.add_argument("--max-qids", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100, 300])
    parser.add_argument("--num-leaves-grid", type=parse_int_grid, default=parse_int_grid("7,15"))
    parser.add_argument("--learning-rate-grid", type=parse_float_grid, default=parse_float_grid("0.03,0.05"))
    parser.add_argument("--n-estimators-grid", type=parse_int_grid, default=parse_int_grid("60,120"))
    parser.add_argument("--blend-grid", type=parse_float_grid, default=parse_float_grid("0,0.1,0.2,0.35"))
    parser.add_argument("--hard-query-weight", type=float, default=3.0)
    parser.add_argument("--tail-positive-weight", type=float, default=2.0)
    parser.add_argument("--hard-rescue-positive-weight", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--metrics-output", default="results/metrics/evidence_unit_hypergraph_v6_sample100.json")
    parser.add_argument("--table-output", default="results/tables/evidence_unit_hypergraph_v6_sample100.md")
    parser.add_argument("--csv-output", default="results/tables/evidence_unit_hypergraph_v6_sample100.csv")
    parser.add_argument("--predictions-output", default="outputs/rerank/evidence_unit_hypergraph_v6_sample100.jsonl")
    return parser.parse_args()


def hard_subset_qids(
    predictions_by_qid: dict[str, list[dict[str, Any]]],
    qrels_by_qid: dict[str, dict[str, float]],
    *,
    pool_k: int,
) -> set[str]:
    hard: set[str] = set()
    for qid, rows in predictions_by_qid.items():
        gold = set(qrels_by_qid.get(str(qid), {}))
        if not gold:
            continue
        top_pool = {str(row["passage_id"]) for row in rows[:pool_k]}
        top10 = {str(row["passage_id"]) for row in rows[:10]}
        if gold & top_pool and not (gold & top10):
            hard.add(str(qid))
    return hard


def row_weights(
    meta: list[dict[str, Any]],
    labels: np.ndarray,
    hard_qids: set[str],
    *,
    hard_query_weight: float,
    tail_positive_weight: float,
    hard_rescue_positive_weight: float,
) -> np.ndarray:
    weights = np.ones(len(meta), dtype=np.float64)
    for idx, item in enumerate(meta):
        if str(item["qid"]) in hard_qids:
            weights[idx] *= hard_query_weight
        if labels[idx] > 0 and int(item.get("base_rank", 999999)) > 10:
            weights[idx] *= tail_positive_weight
        if labels[idx] > 0 and float(item["features"].get("hard_rescue_score", 0.0)) > 0:
            weights[idx] *= hard_rescue_positive_weight
    return weights


def train_and_eval_v6(
    *,
    label: str,
    feature_names: list[str],
    feature_rows: dict[str, list[dict[str, Any]]],
    splits: dict[str, set[str]],
    qrels: list[dict[str, Any]],
    qrels_by_qid: dict[str, dict[str, float]],
    hard_qids: set[str],
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_x, train_y, train_groups, train_meta = feature_matrix(
        feature_rows, splits["train"], qrels_by_qid, feature_names
    )
    if int(train_y.sum()) == 0:
        raise ValueError(f"No positive training labels for {label}.")
    train_weights = row_weights(
        train_meta,
        train_y,
        hard_qids & splits["train"],
        hard_query_weight=args.hard_query_weight,
        tail_positive_weight=args.tail_positive_weight,
        hard_rescue_positive_weight=args.hard_rescue_positive_weight,
    )
    validation_qrels = [row for row in qrels if str(row["question_id"]) in splits["validation"]]
    validation_hard_qids = hard_qids & splits["validation"]
    validation_hard_qrels = [row for row in qrels if str(row["question_id"]) in validation_hard_qids]
    test_qrels = [row for row in qrels if str(row["question_id"]) in splits["test"]]
    all_qrels = [row for row in qrels if str(row["question_id"]) in splits["all"]]
    hard_qrels = [row for row in qrels if str(row["question_id"]) in hard_qids]

    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for num_leaves in args.num_leaves_grid:
        for learning_rate in args.learning_rate_grid:
            for n_estimators in args.n_estimators_grid:
                model = make_lambdamart_ranker(
                    seed=seed,
                    num_leaves=int(num_leaves),
                    learning_rate=float(learning_rate),
                    n_estimators=int(n_estimators),
                )
                model.fit(train_x, train_y, group=train_groups, sample_weight=train_weights)
                for blend_weight in args.blend_grid:
                    validation_predictions = score_model(
                        model,
                        feature_rows,
                        splits["validation"],
                        qrels_by_qid,
                        feature_names,
                        top_k=args.top_k,
                        blend_weight=float(blend_weight),
                        retriever_name=f"{label}_validation",
                    )
                    metrics = evaluate_retrieval(validation_qrels, validation_predictions, sorted(set(args.ks)))
                    validation_hard_predictions = [
                        row for row in validation_predictions if str(row["question_id"]) in validation_hard_qids
                    ]
                    hard_metrics = (
                        evaluate_retrieval(validation_hard_qrels, validation_hard_predictions, sorted(set(args.ks)))
                        if validation_hard_qrels
                        else {}
                    )
                    trial = {
                        "num_leaves": int(num_leaves),
                        "learning_rate": float(learning_rate),
                        "n_estimators": int(n_estimators),
                        "blend_weight": float(blend_weight),
                        "validation_hard_recall@10": float(hard_metrics.get("recall@10", 0.0)),
                        "validation_hard_ndcg@10": float(hard_metrics.get("ndcg@10", 0.0)),
                        "validation_hard_mrr@10": float(hard_metrics.get("mrr@10", 0.0)),
                        "validation_mrr@10": float(metrics.get("mrr@10", 0.0)),
                        "validation_recall@10": float(metrics.get("recall@10", 0.0)),
                        "validation_ndcg@10": float(metrics.get("ndcg@10", 0.0)),
                    }
                    trials.append(trial)
                    key = (
                        trial["validation_hard_recall@10"] if validation_hard_qrels else 0.0,
                        trial["validation_hard_ndcg@10"] if validation_hard_qrels else 0.0,
                        trial["validation_recall@10"],
                        trial["validation_ndcg@10"],
                        trial["validation_mrr@10"],
                        -trial["blend_weight"],
                    )
                    if best is None or key > best["key"]:
                        best = {"key": key, "trial": trial}
    assert best is not None

    final_x, final_y, final_groups, final_meta = feature_matrix(
        feature_rows, splits["train"] | splits["validation"], qrels_by_qid, feature_names
    )
    final_weights = row_weights(
        final_meta,
        final_y,
        hard_qids & (splits["train"] | splits["validation"]),
        hard_query_weight=args.hard_query_weight,
        tail_positive_weight=args.tail_positive_weight,
        hard_rescue_positive_weight=args.hard_rescue_positive_weight,
    )
    selected = best["trial"]
    final_model = make_lambdamart_ranker(
        seed=seed,
        num_leaves=int(selected["num_leaves"]),
        learning_rate=float(selected["learning_rate"]),
        n_estimators=int(selected["n_estimators"]),
    )
    final_model.fit(final_x, final_y, group=final_groups, sample_weight=final_weights)
    blend = float(selected["blend_weight"])
    all_predictions = score_model(
        final_model,
        feature_rows,
        splits["all"],
        qrels_by_qid,
        feature_names,
        top_k=args.top_k,
        blend_weight=blend,
        retriever_name=label,
    )
    test_predictions = [row for row in all_predictions if str(row["question_id"]) in splits["test"]]
    hard_predictions = [row for row in all_predictions if str(row["question_id"]) in hard_qids]

    diagnostics = {
        "label": label,
        "feature_names": feature_names,
        "selected": selected,
        "top_trials": sorted(
            trials,
            key=lambda row: (
                -row["validation_hard_recall@10"],
                -row["validation_hard_ndcg@10"],
                -row["validation_recall@10"],
                -row["validation_ndcg@10"],
                -row["validation_mrr@10"],
            ),
        )[:10],
        "all_sample_metrics": evaluate_retrieval(all_qrels, all_predictions, sorted(set(args.ks))),
        "test_metrics": evaluate_retrieval(test_qrels, test_predictions, sorted(set(args.ks))),
        "hard_subset_metrics": evaluate_retrieval(hard_qrels, hard_predictions, sorted(set(args.ks))) if hard_qrels else {},
        "feature_importance": sorted(
            [
                {"feature": name, "importance": float(value)}
                for name, value in zip(feature_names, final_model.feature_importances_, strict=False)
            ],
            key=lambda row: float(row["importance"]),
            reverse=True,
        ),
        "train_rows": int(train_x.shape[0]),
        "train_positive_rows": int(train_y.sum()),
        "train_hard_qids": sorted(hard_qids & splits["train"], key=qid_sort_key),
        "final_train_rows": int(final_x.shape[0]),
        "final_train_positive_rows": int(final_y.sum()),
    }
    return all_predictions, diagnostics


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_row(method: str, metrics: dict[str, Any], base: dict[str, Any], hard: dict[str, Any]) -> dict[str, str]:
    return {
        "method": method,
        "recall@10": f"{float(metrics.get('recall@10', 0.0)):.4f}",
        "mrr@10": f"{float(metrics.get('mrr@10', 0.0)):.4f}",
        "ndcg@10": f"{float(metrics.get('ndcg@10', 0.0)):.4f}",
        "recall@100": f"{float(metrics.get('recall@100', 0.0)):.4f}",
        "hard_recall@10": f"{float(hard.get('recall@10', 0.0)):.4f}" if hard else "",
        "hard_ndcg@10": f"{float(hard.get('ndcg@10', 0.0)):.4f}" if hard else "",
        "delta_recall@10": f"{float(metrics.get('recall@10', 0.0)) - float(base.get('recall@10', 0.0)):+.4f}",
        "delta_ndcg@10": f"{float(metrics.get('ndcg@10', 0.0)) - float(base.get('ndcg@10', 0.0)):+.4f}",
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(seed)
    paths = config["paths"]

    qrels_path = args.qrels or paths.get("qrels", "data/processed/bioasq_qrels.jsonl")
    corpus_path = args.corpus or paths.get("corpus", "data/processed/bioasq_corpus.jsonl")
    question_entities_path = args.question_entities or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    passage_entities_path = args.passage_entities or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")
    question_mesh_path = args.question_mesh or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")
    passage_mesh_path = args.passage_mesh or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")
    mesh_hierarchy_path = args.mesh_hierarchy or paths.get("mesh_hierarchy", "data/external_knowledge/mesh_hierarchy_2026.jsonl")
    pubmed_metadata_path = args.pubmed_metadata or paths.get("pubmed_mesh", "data/external_knowledge/pubmed_mesh.jsonl")

    qrels = read_jsonl(qrels_path)
    qrels_by_qid = group_qrels(qrels)
    qids, predictions_by_qid = select_prediction_rows(
        args.predictions,
        qrels_by_qid,
        top_k=args.top_k,
        max_qids=args.max_qids,
        qid_file=args.qid_file,
        hard_only=False,
    )
    if len(qids) < 5:
        raise RuntimeError("Need at least 5 qids for v6 diagnostics.")
    splits = split_qids(qids)
    selected_qids = set(qids)
    selected_qrels = [row for row in qrels if str(row["question_id"]) in selected_qids]
    base_predictions_all = flatten_predictions(predictions_by_qid, selected_qids, args.top_k)
    base_metrics = evaluate_retrieval(selected_qrels, base_predictions_all, sorted(set(args.ks)))
    hard_qids = hard_subset_qids(predictions_by_qid, qrels_by_qid, pool_k=min(args.top_k, 100))
    hard_qrels = [row for row in qrels if str(row["question_id"]) in hard_qids]
    hard_base_predictions = [row for row in base_predictions_all if str(row["question_id"]) in hard_qids]
    hard_base_metrics = evaluate_retrieval(hard_qrels, hard_base_predictions, sorted(set(args.ks))) if hard_qrels else {}

    question_entities = entity_map(read_jsonl(question_entities_path), "question_id")
    passage_entities = entity_map(read_jsonl(passage_entities_path), "passage_id")
    pubtator_stats = {"enabled": bool(args.use_pubtator_concepts)}
    if args.use_pubtator_concepts:
        question_pubtator = load_pubtator_entity_map(args.question_pubtator_concepts, "question_id")
        passage_pubtator = load_pubtator_entity_map(args.passage_pubtator_concepts, "passage_id")
        pubtator_stats.update(
            {
                "question_rows": len(question_pubtator),
                "passage_rows": len(passage_pubtator),
                "question_concepts": sum(len(rows) for rows in question_pubtator.values()),
                "passage_concepts": sum(len(rows) for rows in passage_pubtator.values()),
            }
        )
        question_entities = merge_entity_maps(question_entities, question_pubtator)
        passage_entities = merge_entity_maps(passage_entities, passage_pubtator)

    question_mesh = mesh_map(read_jsonl(question_mesh_path), "question_id")
    passage_mesh = mesh_map(read_jsonl(passage_mesh_path), "passage_id")
    mesh_hierarchy = load_mesh_hierarchy(read_jsonl(mesh_hierarchy_path))
    corpus = load_map(corpus_path, "passage_id")
    pubmed_metadata = load_pubmed_metadata(pubmed_metadata_path)

    feature_rows = build_evidence_unit_v6_rows(
        predictions_by_qid,
        question_entities=question_entities,
        passage_entities=passage_entities,
        question_mesh=question_mesh,
        passage_mesh=passage_mesh,
        mesh_hierarchy=mesh_hierarchy,
        corpus=corpus,
        pubmed_metadata=pubmed_metadata,
        config=EvidenceUnitConfig(),
    )

    method_predictions: dict[str, list[dict[str, Any]]] = {"source_order": base_predictions_all}
    method_diagnostics: dict[str, Any] = {
        "source_order": {
            "all_sample_metrics": base_metrics,
            "hard_subset_metrics": hard_base_metrics,
        }
    }
    for label in [
        "retrieval_ltr",
        "flat_evidence_unit_ltr",
        "v6_two_layer_hypergraph_ltr",
        "without_topic_hyperedges",
        "without_sufficiency_necessity",
        "without_hard_rescue",
    ]:
        predictions, diagnostics = train_and_eval_v6(
            label=label,
            feature_names=FEATURE_SETS[label],
            feature_rows=feature_rows,
            splits=splits,
            qrels=selected_qrels,
            qrels_by_qid=qrels_by_qid,
            hard_qids=hard_qids,
            args=args,
            seed=seed,
        )
        method_predictions[label] = predictions
        method_diagnostics[label] = diagnostics

    full_predictions = method_predictions["v6_two_layer_hypergraph_ltr"]
    write_jsonl(args.predictions_output, full_predictions)

    rows = [
        metric_row("Source candidate order", base_metrics, base_metrics, hard_base_metrics),
    ]
    for label in [
        "retrieval_ltr",
        "flat_evidence_unit_ltr",
        "v6_two_layer_hypergraph_ltr",
        "without_topic_hyperedges",
        "without_sufficiency_necessity",
        "without_hard_rescue",
    ]:
        diagnostics = method_diagnostics[label]
        rows.append(
            metric_row(
                label,
                diagnostics.get("all_sample_metrics", {}),
                base_metrics,
                diagnostics.get("hard_subset_metrics", {}),
            )
        )
    columns = [
        "method",
        "recall@10",
        "mrr@10",
        "ndcg@10",
        "recall@100",
        "hard_recall@10",
        "hard_ndcg@10",
        "delta_recall@10",
        "delta_ndcg@10",
    ]
    write_csv(args.csv_output, rows, columns)
    write_markdown(args.table_output, rows, columns)

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "method": "evidence-unit hypergraph v6",
        "method_changes": [
            "Dou-style two-layer evidence hyperedges and topic hyperedges",
            "hard-subset-aware LambdaMART sample weighting",
            "CACHE/SHy-inspired sufficiency and necessity support features",
            "PrimeKG-free emphasis on MeSH, PICO-like categories, and evidence units",
        ],
        "seed": seed,
        "selection": {
            "num_qids": len(qids),
            "qids": qids,
            "hard_qids": sorted(hard_qids, key=qid_sort_key),
            "num_hard_qids": len(hard_qids),
            "top_k": args.top_k,
        },
        "split": {key: sorted(value, key=qid_sort_key) for key, value in splits.items()},
        "paths": {
            "predictions": args.predictions,
            "qrels": qrels_path,
            "corpus": corpus_path,
            "question_entities": question_entities_path,
            "passage_entities": passage_entities_path,
            "question_mesh": question_mesh_path,
            "passage_mesh": passage_mesh_path,
            "mesh_hierarchy": mesh_hierarchy_path,
            "pubmed_metadata": pubmed_metadata_path,
        },
        "pubtator": pubtator_stats,
        "base_metrics": base_metrics,
        "hard_base_metrics": hard_base_metrics,
        "method_diagnostics": method_diagnostics,
        "outputs": {
            "metrics": args.metrics_output,
            "table": args.table_output,
            "csv": args.csv_output,
            "predictions": args.predictions_output,
        },
    }
    write_json(args.metrics_output, payload)
    print(
        {
            "metrics": args.metrics_output,
            "table": args.table_output,
            "source_recall@10": base_metrics.get("recall@10"),
            "v6_recall@10": method_diagnostics["v6_two_layer_hypergraph_ltr"]["all_sample_metrics"].get("recall@10"),
            "source_ndcg@10": base_metrics.get("ndcg@10"),
            "v6_ndcg@10": method_diagnostics["v6_two_layer_hypergraph_ltr"]["all_sample_metrics"].get("ndcg@10"),
            "hard_qids": len(hard_qids),
            "hard_v6_recall@10": method_diagnostics["v6_two_layer_hypergraph_ltr"]["hard_subset_metrics"].get("recall@10"),
            "selected": method_diagnostics["v6_two_layer_hypergraph_ltr"]["selected"],
        }
    )


if __name__ == "__main__":
    main()
