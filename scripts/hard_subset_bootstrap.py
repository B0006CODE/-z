from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import group_predictions, group_qrels, dcg
from src.utils import read_jsonl


def per_query_recall(gold: dict[str, float], rows: list[dict[str, Any]], k: int) -> float:
    ranked_ids = {str(row["passage_id"]) for row in rows[:k]}
    gold_ids = set(gold)
    hits = gold_ids & ranked_ids
    return len(hits) / len(gold_ids) if gold_ids else 0.0


def per_query_mrr(gold: dict[str, float], rows: list[dict[str, Any]], k: int) -> float:
    for rank, row in enumerate(rows[:k], start=1):
        if str(row["passage_id"]) in gold:
            return 1.0 / rank
    return 0.0


def per_query_ndcg(gold: dict[str, float], rows: list[dict[str, Any]], k: int) -> float:
    gains = [gold.get(str(row["passage_id"]), 0.0) for row in rows[:k]]
    ideal = sorted(gold.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0


def main() -> None:
    qrels_path = "data/processed/bioasq_qrels.jsonl"
    hybrid_path = "outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl"
    semantic_path = "outputs/rerank/kch_medrank_enhanced_bioasq_v2_semantic_no_hypergraph_ltr_test_top100.jsonl"
    full_path = "outputs/rerank/kch_medrank_enhanced_bioasq_v2_full_kch_medrank_test_top100.jsonl"

    qrels = read_jsonl(qrels_path)
    qrels_by_qid = group_qrels(qrels)

    hybrid_preds = read_jsonl(hybrid_path)
    hybrid_by_qid = group_predictions(hybrid_preds)

    semantic_preds = read_jsonl(semantic_path)
    semantic_by_qid = group_predictions(semantic_preds)

    full_preds = read_jsonl(full_path)
    full_by_qid = group_predictions(full_preds)

    test_qids = {qid for qid in hybrid_by_qid if int(qid) % 5 == 4}

    hard_qids = set()
    for qid in sorted(test_qids):
        gold = set(qrels_by_qid.get(qid, {}))
        ranked = hybrid_by_qid.get(qid, [])
        top10 = {str(row["passage_id"]) for row in ranked[:10]}
        top100 = {str(row["passage_id"]) for row in ranked[:100]}
        if gold & top100 and not (gold & top10):
            hard_qids.add(qid)

    print(f"Hard subset size: {len(hard_qids)}")

    semantic_recall = []
    full_recall = []
    for qid in sorted(hard_qids):
        gold = qrels_by_qid.get(qid, {})
        sem_rows = semantic_by_qid.get(qid, [])
        full_rows = full_by_qid.get(qid, [])
        semantic_recall.append(per_query_recall(gold, sem_rows, 10))
        full_recall.append(per_query_recall(gold, full_rows, 10))

    sem_arr = np.array(semantic_recall)
    full_arr = np.array(full_recall)
    delta = full_arr - sem_arr
    n = len(delta)

    rng = np.random.default_rng(42)
    n_boot = 10000
    boot_deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas[i] = float(np.mean(delta[idx]))

    p_lower = (float(np.sum(boot_deltas <= 0.0)) + 1.0) / (n_boot + 1.0)
    p_upper = (float(np.sum(boot_deltas >= 0.0)) + 1.0) / (n_boot + 1.0)
    p_two_sided = min(1.0, 2.0 * min(p_lower, p_upper))
    ci_lower = float(np.quantile(boot_deltas, 0.025))
    ci_upper = float(np.quantile(boot_deltas, 0.975))

    print(f"\nHard Subset Paired Bootstrap: Full KCH-MedRank vs Semantic (no hypergraph)")
    print(f"{'Metric':<12} {'Semantic':>10} {'Full':>10} {'Delta':>10} {'CI Low':>10} {'CI High':>10} {'p-value':>10}")
    print("-" * 82)
    print(f"{'Recall@10':<12} {float(np.mean(sem_arr)):>10.4f} {float(np.mean(full_arr)):>10.4f} {float(np.mean(delta)):>10.4f} {ci_lower:>10.4f} {ci_upper:>10.4f} {p_two_sided:>10.4f}")

    if p_two_sided < 0.05:
        print("\n*** SIGNIFICANT at p < 0.05 ***")
    elif p_two_sided < 0.10:
        print("\n* TREND (p < 0.10)")
    else:
        print("\nNOT SIGNIFICANT")


if __name__ == "__main__":
    from typing import Any
    main()
