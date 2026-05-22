"""Quick bootstrap: v2 KCH-MedRank vs old MedCPT Cross-Encoder on same test split"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import dcg, group_predictions, group_qrels
from src.utils import read_jsonl


def per_query_recall(gold, rows, k):
    ranked = {str(r["passage_id"]) for r in rows[:k]}
    return len(set(gold) & ranked) / len(gold) if gold else 0.0


def per_query_mrr(gold, rows, k):
    for rank, r in enumerate(rows[:k], start=1):
        if str(r["passage_id"]) in gold:
            return 1.0 / rank
    return 0.0


def per_query_ndcg(gold, rows, k):
    gains = [gold.get(str(r["passage_id"]), 0.0) for r in rows[:k]]
    ideal = sorted(gold.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0


def main():
    qrels_by_qid = group_qrels(read_jsonl("data/processed/bioasq_qrels.jsonl"))
    hybrid_by_qid = group_predictions(read_jsonl("outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl"))

    kch_v2 = group_predictions(read_jsonl("outputs/rerank/kch_medrank_enhanced_bioasq_v2_full_kch_medrank_test_top100.jsonl"))
    medcpt = read_jsonl("outputs/rerank/medcpt_cross_encoder_enhanced_bioasq_test_top100.jsonl")
    medcpt_by = group_predictions(medcpt) if medcpt else {}

    test_qids = sorted({qid for qid in hybrid_by_qid if int(qid) % 5 == 4})

    if not medcpt_by:
        print("WARNING: Cross-encoder predictions not found. Using approximate values.")
        return

    rng = np.random.default_rng(42)
    n_boot = 10000

    for metric_name, metric_fn in [("Recall", per_query_recall), ("MRR", per_query_mrr), ("NDCG", per_query_ndcg)]:
        kch_vals = np.array([metric_fn(qrels_by_qid.get(qid, {}), kch_v2.get(qid, []), 10) for qid in test_qids])
        med_vals = np.array([metric_fn(qrels_by_qid.get(qid, {}), medcpt_by.get(qid, []), 10) for qid in test_qids])
        delta = kch_vals - med_vals
        n = len(delta)
        boot = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot[i] = float(np.mean(delta[idx]))
        p_lower = (float(np.sum(boot <= 0)) + 1) / (n_boot + 1)
        p_upper = (float(np.sum(boot >= 0)) + 1) / (n_boot + 1)
        p = min(1.0, 2 * min(p_lower, p_upper))
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")
        print(f"{metric_name}@10: KCH={float(np.mean(kch_vals)):.4f} CE={float(np.mean(med_vals)):.4f} Δ={float(np.mean(delta)):+.4f} CI=[{float(np.quantile(boot,0.025)):+.4f},{float(np.quantile(boot,0.975)):+.4f}] p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
