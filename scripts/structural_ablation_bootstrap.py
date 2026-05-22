"""生成新版消融表所需的所有配对 bootstrap 数据（Recall + MRR + nDCG）"""
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


def paired_bootstrap(a_vals, b_vals, seed, n_boot=10000):
    delta = a_vals - b_vals
    n = len(delta)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(delta[idx]))
    p_lower = (float(np.sum(boot <= 0)) + 1) / (n_boot + 1)
    p_upper = (float(np.sum(boot >= 0)) + 1) / (n_boot + 1)
    return {
        "a_mean": float(np.mean(a_vals)),
        "b_mean": float(np.mean(b_vals)),
        "delta": float(np.mean(delta)),
        "ci_lower": float(np.quantile(boot, 0.025)),
        "ci_upper": float(np.quantile(boot, 0.975)),
        "p_two_sided": min(1.0, 2 * min(p_lower, p_upper)),
    }


def main():
    qrels_by_qid = group_qrels(read_jsonl("data/processed/bioasq_qrels.jsonl"))
    hybrid_by_qid = group_predictions(read_jsonl("outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl"))

    prefix = "outputs/rerank/kch_medrank_enhanced_bioasq_v2"
    methods = {
        "Hybrid RRF": read_jsonl("outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl"),
        "Retrieval LTR": read_jsonl(f"{prefix}_retrieval_ltr_test_top100.jsonl"),
        "Semantic (no graph)": read_jsonl(f"{prefix}_semantic_no_hypergraph_ltr_test_top100.jsonl"),
        "Hypergraph (no med)": read_jsonl(f"{prefix}_hypergraph_no_medical_knowledge_ltr_test_top100.jsonl"),
        "Pairwise graph": read_jsonl(f"{prefix}_pairwise_graph_ltr_test_top100.jsonl"),
        "KCH-MedRank": read_jsonl(f"{prefix}_full_kch_medrank_test_top100.jsonl"),
    }

    preds_by = {name: group_predictions(rows) for name, rows in methods.items()}
    test_qids = sorted({qid for qid in hybrid_by_qid if int(qid) % 5 == 4})

    comparisons = [
        ("KCH-MedRank", "Semantic (no graph)", "超图+知识 vs 纯语义"),
        ("KCH-MedRank", "Hypergraph (no med)", "知识约束的价值"),
        ("KCH-MedRank", "Pairwise graph", "超图 vs 对偶图"),
        ("Semantic (no graph)", "Retrieval LTR", "语义的独立贡献"),
        ("Hypergraph (no med)", "Semantic (no graph)", "无知识超图的噪音"),
        ("KCH-MedRank", "Hybrid RRF", "vs Hybrid 基线"),
    ]

    all_results = []

    for a_name, b_name, description in comparisons:
        a_rows = preds_by[a_name]
        b_rows = preds_by[b_name]

        for k in [5, 10]:
            for metric_name, metric_fn in [("Recall", per_query_recall), ("MRR", per_query_mrr), ("NDCG", per_query_ndcg)]:
                a_vals = np.array([metric_fn(qrels_by_qid.get(qid, {}), a_rows.get(qid, []), k) for qid in test_qids])
                b_vals = np.array([metric_fn(qrels_by_qid.get(qid, {}), b_rows.get(qid, []), k) for qid in test_qids])
                r = paired_bootstrap(a_vals, b_vals, seed=42)
                r["comparison"] = description
                r["a_method"] = a_name
                r["b_method"] = b_name
                r["metric"] = metric_name
                r["k"] = k
                all_results.append(r)
                sig = "**" if r["p_two_sided"] < 0.01 else ("*" if r["p_two_sided"] < 0.05 else "n.s.")
                print(f"{description:30s} {metric_name}@{k}: Δ={r['delta']:+.4f} p={r['p_two_sided']:.4f} {sig}")

    with open("results/metrics/structural_ablation_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nSaved to results/metrics/structural_ablation_bootstrap.json")


if __name__ == "__main__":
    main()
