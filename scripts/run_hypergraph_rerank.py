from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.rerank.hypergraph import build_feature_rows, entity_map, mesh_map, relations_map, rerank_from_features
from src.utils import load_config, read_jsonl, set_seed, write_json, write_jsonl


def parse_float_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Grid must contain at least one float value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank retrieved candidates with local hypergraph diffusion features.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--predictions", default="outputs/retrieval/hybrid_full_top100.jsonl")
    parser.add_argument("--question-entities", default=None)
    parser.add_argument("--passage-entities", default=None)
    parser.add_argument("--question-mesh", default=None)
    parser.add_argument("--passage-mesh", default=None)
    parser.add_argument("--disable-mesh", action="store_true", help="Do not load MeSH features; used for remove-MeSH ablations.")
    parser.add_argument("--relations", default=None)
    parser.add_argument("--disable-relations", action="store_true", help="Do not load PrimeKG relation features.")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--structure",
        choices=["knowledge_hypergraph", "no_knowledge_hypergraph", "pairwise_graph"],
        default="knowledge_hypergraph",
        help="Local reranking structure for ablation experiments.",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--max-passage-entities", type=int, default=48)
    parser.add_argument("--max-passage-mesh", type=int, default=32)
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--hypergraph-weight", type=float, default=0.0)
    parser.add_argument("--entity-weight", type=float, default=0.0)
    parser.add_argument("--mesh-weight", type=float, default=0.0)
    parser.add_argument("--relation-weight", type=float, default=0.0)
    parser.add_argument("--tune-weights", action="store_true", help="Tune weights on a deterministic validation split.")
    parser.add_argument("--hypergraph-grid", type=parse_float_grid, default=parse_float_grid("0,0.02,0.05,0.1,0.2,0.4"))
    parser.add_argument("--entity-grid", type=parse_float_grid, default=parse_float_grid("0,0.02,0.05,0.1,0.2"))
    parser.add_argument("--mesh-grid", type=parse_float_grid, default=parse_float_grid("0,0.02,0.05,0.1,0.2"))
    parser.add_argument("--relation-grid", type=parse_float_grid, default=parse_float_grid("0,0.02,0.05"))
    parser.add_argument("--validation-modulo", type=int, default=5)
    parser.add_argument("--validation-remainder", type=int, default=0)
    parser.add_argument(
        "--target-split",
        choices=["all", "validation", "test"],
        default="all",
        help="Which deterministic split to write. Use test with --only-predicted-qids evaluation for an unbiased check.",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    return parser.parse_args()


def qid_bucket(qid: str, modulo: int) -> int:
    if qid.isdigit():
        return int(qid) % modulo
    return sum(ord(char) for char in qid) % modulo


def split_qids(qids: list[str], modulo: int, remainder: int) -> tuple[set[str], set[str]]:
    validation = {qid for qid in qids if qid_bucket(qid, modulo) == remainder}
    test = set(qids) - validation
    return validation, test


def filter_feature_rows(features_by_qid: dict[str, list[dict[str, Any]]], keep_qids: set[str]) -> dict[str, list[dict[str, Any]]]:
    return {qid: rows for qid, rows in features_by_qid.items() if qid in keep_qids}


def filter_qrels(qrels: list[dict[str, Any]], keep_qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in qrels if str(row["question_id"]) in keep_qids]


def tune_weights(
    features_by_qid: dict[str, list[dict[str, Any]]],
    qrels: list[dict[str, Any]],
    validation_qids: set[str],
    *,
    top_k: int,
    ks: list[int],
    base_weight: float,
    hypergraph_grid: list[float],
    entity_grid: list[float],
    mesh_grid: list[float],
    relation_grid: list[float],
) -> dict[str, Any]:
    validation_features = filter_feature_rows(features_by_qid, validation_qids)
    validation_qrels = filter_qrels(qrels, validation_qids)
    best: dict[str, Any] | None = None
    trials = []
    for hypergraph_weight in hypergraph_grid:
        for entity_weight in entity_grid:
            for mesh_weight in mesh_grid:
                for relation_weight in relation_grid:
                    predictions = rerank_from_features(
                        validation_features,
                        top_k=top_k,
                        base_weight=base_weight,
                        hypergraph_weight=hypergraph_weight,
                        entity_weight=entity_weight,
                        mesh_weight=mesh_weight,
                        relation_weight=relation_weight,
                        retriever_name="local_hypergraph_rerank_validation",
                    )
                    metrics = evaluate_retrieval(validation_qrels, predictions, sorted(set(ks)))
                    trial = {
                        "base_weight": base_weight,
                        "hypergraph_weight": hypergraph_weight,
                        "entity_weight": entity_weight,
                        "mesh_weight": mesh_weight,
                        "relation_weight": relation_weight,
                        "mrr@10": metrics.get("mrr@10", 0.0),
                        "recall@10": metrics.get("recall@10", 0.0),
                        "recall@5": metrics.get("recall@5", 0.0),
                        "ndcg@10": metrics.get("ndcg@10", 0.0),
                    }
                    trials.append(trial)
                    key = (
                        trial["mrr@10"],
                        trial["recall@10"],
                        trial["ndcg@10"],
                        -hypergraph_weight,
                        -entity_weight,
                        -mesh_weight,
                        -relation_weight,
                    )
                    if best is None or key > best["key"]:
                        best = {"key": key, "trial": trial}

    assert best is not None
    return {
        "selected": best["trial"],
        "trials": sorted(trials, key=lambda row: (-row["mrr@10"], -row["recall@10"], -row["ndcg@10"]))[:20],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    paths = config["paths"]

    question_entities_path = args.question_entities or paths.get("question_entities", "data/processed/bioasq_question_entities.jsonl")
    passage_entities_path = args.passage_entities or paths.get("passage_entities", "data/processed/bioasq_passage_entities.jsonl")
    question_mesh_path = args.question_mesh or paths.get("question_mesh", "data/processed/bioasq_question_mesh.jsonl")
    passage_mesh_path = args.passage_mesh or paths.get("passage_mesh", "data/processed/bioasq_passage_mesh.jsonl")
    relations_path = args.relations or paths.get("primekg_relations", "data/external_knowledge/primekg_project_relations.jsonl")
    qrels_path = args.qrels or paths.get("qrels", "data/processed/bioasq_qrels.jsonl")
    output_path = args.output or paths.get("hypergraph_predictions", "outputs/rerank/hypergraph_full_top100.jsonl")
    metrics_output_path = args.metrics_output or paths.get("hypergraph_metrics", "results/metrics/hypergraph_full_top100_metrics.json")

    predictions = read_jsonl(args.predictions)
    qrels = read_jsonl(qrels_path)
    question_entities = entity_map(read_jsonl(question_entities_path), "question_id")
    passage_entities = entity_map(read_jsonl(passage_entities_path), "passage_id")
    question_mesh = (
        {}
        if args.disable_mesh
        else mesh_map(read_jsonl(question_mesh_path), "question_id") if Path(question_mesh_path).exists() else {}
    )
    passage_mesh = (
        {}
        if args.disable_mesh
        else mesh_map(read_jsonl(passage_mesh_path), "passage_id") if Path(passage_mesh_path).exists() else {}
    )
    entity_relations = (
        {}
        if args.disable_relations
        else relations_map(read_jsonl(relations_path)) if Path(relations_path).exists() else {}
    )

    features_by_qid = build_feature_rows(
        predictions,
        question_entities,
        passage_entities,
        question_mesh,
        passage_mesh,
        entity_relations,
        structure=args.structure,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        iterations=args.iterations,
        damping=args.damping,
        max_passage_entities=args.max_passage_entities,
        max_passage_mesh=args.max_passage_mesh,
    )
    all_qids = sorted(features_by_qid)
    validation_qids, test_qids = split_qids(all_qids, args.validation_modulo, args.validation_remainder)

    selected = {
        "base_weight": args.base_weight,
        "hypergraph_weight": args.hypergraph_weight,
        "entity_weight": args.entity_weight,
        "mesh_weight": args.mesh_weight,
        "relation_weight": args.relation_weight,
    }
    tuning: dict[str, Any] | None = None
    if args.tune_weights:
        tuning = tune_weights(
            features_by_qid,
            qrels,
            validation_qids,
            top_k=args.top_k,
            ks=args.ks,
            base_weight=args.base_weight,
            hypergraph_grid=args.hypergraph_grid,
            entity_grid=args.entity_grid,
            mesh_grid=args.mesh_grid,
            relation_grid=args.relation_grid,
        )
        selected = {
            "base_weight": float(tuning["selected"]["base_weight"]),
            "hypergraph_weight": float(tuning["selected"]["hypergraph_weight"]),
            "entity_weight": float(tuning["selected"]["entity_weight"]),
            "mesh_weight": float(tuning["selected"]["mesh_weight"]),
            "relation_weight": float(tuning["selected"]["relation_weight"]),
        }

    if args.target_split == "validation":
        target_qids = validation_qids
    elif args.target_split == "test":
        target_qids = test_qids
    else:
        target_qids = set(all_qids)

    target_features = filter_feature_rows(features_by_qid, target_qids)
    reranked = rerank_from_features(
        target_features,
        top_k=args.top_k,
        base_weight=selected["base_weight"],
        hypergraph_weight=selected["hypergraph_weight"],
        entity_weight=selected["entity_weight"],
        mesh_weight=selected["mesh_weight"],
        relation_weight=selected["relation_weight"],
        retriever_name=args.structure,
    )
    write_jsonl(output_path, reranked)

    target_qrels = filter_qrels(qrels, target_qids)
    metrics = evaluate_retrieval(target_qrels, reranked, sorted(set(args.ks)))
    metrics.update(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "qrels": qrels_path,
            "predictions": output_path,
            "source_predictions": args.predictions,
            "question_mesh": question_mesh_path,
            "passage_mesh": passage_mesh_path,
            "mesh_enabled": not args.disable_mesh,
            "relations": relations_path,
            "relations_enabled": not args.disable_relations,
            "structure": args.structure,
            "target_split": args.target_split,
            "num_validation_qids": len(validation_qids),
            "num_test_qids": len(test_qids),
            "selected_weights": selected,
            "tuning": tuning,
        }
    )
    write_json(metrics_output_path, metrics)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "predictions": args.predictions,
        "question_entities": question_entities_path,
        "passage_entities": passage_entities_path,
        "question_mesh": question_mesh_path,
        "passage_mesh": passage_mesh_path,
        "mesh_enabled": not args.disable_mesh,
        "relations": relations_path,
        "relations_enabled": not args.disable_relations,
        "output": output_path,
        "metrics_output": metrics_output_path,
        "structure": args.structure,
        "top_k": args.top_k,
        "iterations": args.iterations,
        "damping": args.damping,
        "target_split": args.target_split,
        "selected_weights": selected,
        "num_questions": len(target_features),
        "num_predictions": len(reranked),
    }
    write_json(Path(paths["logs_dir"]) / "run_hypergraph_rerank_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
