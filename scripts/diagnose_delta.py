from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import group_predictions, group_qrels
from src.utils import read_jsonl


def per_query_recall(gold: dict[str, float], rows: list[dict[str, Any]], k: int) -> float:
    ranked_ids = {str(row["passage_id"]) for row in rows[:k]}
    gold_ids = set(gold)
    hits = gold_ids & ranked_ids
    return len(hits) / len(gold_ids) if gold_ids else 0.0


def main() -> None:
    qrels_path = "data/processed/bioasq_qrels.jsonl"
    hybrid_path = "outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl"
    semantic_path = "outputs/rerank/kch_medrank_enhanced_bioasq_v2_semantic_no_hypergraph_ltr_test_top100.jsonl"
    full_path = "outputs/rerank/kch_medrank_enhanced_bioasq_v2_full_kch_medrank_test_top100.jsonl"
    mesh_ablation_path = "outputs/rerank/kch_medrank_enhanced_bioasq_v2_remove_mesh_hierarchy_test_top100.jsonl"
    hg_ablation_path = "outputs/rerank/kch_medrank_enhanced_bioasq_v2_remove_hypergraph_test_top100.jsonl"

    qrels_by_qid = group_qrels(read_jsonl(qrels_path))
    hybrid_by_qid = group_predictions(read_jsonl(hybrid_path))
    semantic_by_qid = group_predictions(read_jsonl(semantic_path))
    full_by_qid = group_predictions(read_jsonl(full_path))

    test_qids = {qid for qid in hybrid_by_qid if int(qid) % 5 == 4}

    hard_qids = set()
    for qid in sorted(test_qids):
        gold = set(qrels_by_qid.get(qid, {}))
        ranked = hybrid_by_qid.get(qid, [])
        top10 = {str(row["passage_id"]) for row in ranked[:10]}
        top100 = {str(row["passage_id"]) for row in ranked[:100]}
        if gold & top100 and not (gold & top10):
            hard_qids.add(qid)

    n_total = 0
    n_full_win = 0
    n_sem_win = 0
    n_tie = 0
    n_full_total = 0
    n_sem_total = 0
    n_none = 0

    for qid in sorted(test_qids):
        gold = qrels_by_qid.get(qid, {})
        sem_r = per_query_recall(gold, semantic_by_qid.get(qid, []), 10)
        full_r = per_query_recall(gold, full_by_qid.get(qid, []), 10)
        n_total += 1
        if full_r > sem_r:
            n_full_win += 1
            n_full_total += 1
        elif sem_r > full_r:
            n_sem_win += 1
            n_sem_total += 1
        elif full_r > 0:
            n_tie += 1

    for qid in sorted(test_qids):
        gold = qrels_by_qid.get(qid, {})
        sem_r = per_query_recall(gold, semantic_by_qid.get(qid, []), 10)
        full_r = per_query_recall(gold, full_by_qid.get(qid, []), 10)
        if sem_r == 0 and full_r == 0:
            n_none += 1

    print(f"=== Full test set (n={n_total}) ===")
    print(f"Full KCH-MedRank wins:  {n_full_win:>4d}")
    print(f"Semantic-only wins:     {n_sem_win:>4d}")
    print(f"Tie (both have recall): {n_tie:>4d}")
    print(f"Both zero recall:       {n_none:>4d}")

    n_hard_total = 0
    n_hard_full_win = 0
    n_hard_sem_win = 0
    n_hard_tie = 0
    n_hard_none = 0

    for qid in sorted(hard_qids):
        gold = qrels_by_qid.get(qid, {})
        sem_r = per_query_recall(gold, semantic_by_qid.get(qid, []), 10)
        full_r = per_query_recall(gold, full_by_qid.get(qid, []), 10)
        n_hard_total += 1
        if full_r > sem_r:
            n_hard_full_win += 1
        elif sem_r > full_r:
            n_hard_sem_win += 1
        elif full_r > 0:
            n_hard_tie += 1
        else:
            n_hard_none += 1

    print(f"\n=== Hard subset (n={n_hard_total}) ===")
    print(f"Full KCH-MedRank wins:  {n_hard_full_win:>4d}")
    print(f"Semantic-only wins:     {n_hard_sem_win:>4d}")
    print(f"Tie (both have recall): {n_hard_tie:>4d}")
    print(f"Both zero recall:       {n_hard_none:>4d}")

    all_qids = sorted(test_qids)
    n_recall_ceiling = 0
    n_can_recover = 0
    for qid in all_qids:
        gold = qrels_by_qid.get(qid, {})
        sem_r = per_query_recall(gold, semantic_by_qid.get(qid, []), 10)
        if sem_r >= 1.0:
            n_recall_ceiling += 1
        elif sem_r > 0:
            n_can_recover += 1

    print(f"\n=== Ceiling Analysis (all test, n={n_total}) ===")
    print(f"Recall already 100% (can't improve):   {n_recall_ceiling:>4d}")
    print(f"Partial recall (room for improvement):  {n_can_recover:>4d}")
    print(f"Zero recall (can improve):              {n_none:>4d}")

    deltas = []
    for qid in sorted(test_qids):
        gold = qrels_by_qid.get(qid, {})
        sem_r = per_query_recall(gold, semantic_by_qid.get(qid, []), 10)
        full_r = per_query_recall(gold, full_by_qid.get(qid, []), 10)
        deltas.append(full_r - sem_r)
    deltas = np.array(deltas)
    pos = np.sum(deltas > 0)
    neg = np.sum(deltas < 0)
    zero = np.sum(deltas == 0)
    print(f"\n=== Per-question delta distribution ===")
    print(f"Full > Semantic: {pos}")
    print(f"Full = Semantic: {zero}")
    print(f"Full < Semantic: {neg}")
    if pos > 0:
        print(f"Mean positive delta: {np.mean(deltas[deltas > 0]):.4f}")
    if neg > 0:
        print(f"Mean negative delta: {np.mean(deltas[deltas < 0]):.4f}")


if __name__ == "__main__":
    from typing import Any
    main()
