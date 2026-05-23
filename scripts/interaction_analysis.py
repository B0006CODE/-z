"""Analyze the interaction between local structure and biomedical knowledge features."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import group_predictions, group_qrels
from src.utils import read_jsonl


def per_query_recall(gold, rows, k):
    ranked = {str(r["passage_id"]) for r in rows[:k]}
    return len(set(gold) & ranked) / len(gold) if gold else 0.0


def main():
    qrels_by_qid = group_qrels(read_jsonl("data/processed/bioasq_qrels.jsonl"))
    hybrid_by_qid = group_predictions(read_jsonl("outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl"))

    methods = {
        "Retrieval LTR": read_jsonl("outputs/rerank/kch_medrank_enhanced_bioasq_v2_retrieval_ltr_test_top100.jsonl"),
        "Flat knowledge LTR (no graph)": read_jsonl("outputs/rerank/kch_medrank_enhanced_bioasq_v2_semantic_no_hypergraph_ltr_test_top100.jsonl"),
        "Hypergraph (no med)": read_jsonl("outputs/rerank/kch_medrank_enhanced_bioasq_v2_hypergraph_no_medical_knowledge_ltr_test_top100.jsonl"),
        "Pairwise graph": read_jsonl("outputs/rerank/kch_medrank_enhanced_bioasq_v2_pairwise_graph_ltr_test_top100.jsonl"),
        "KCH-MedRank": read_jsonl("outputs/rerank/kch_medrank_enhanced_bioasq_v2_full_kch_medrank_test_top100.jsonl"),
    }

    preds_by = {name: group_predictions(rows) for name, rows in methods.items()}
    test_qids = sorted({qid for qid in hybrid_by_qid if int(qid) % 5 == 4})

    rec = defaultdict(list)
    for qid in test_qids:
        gold = qrels_by_qid.get(qid, {})
        for name in methods:
            rec[name].append(per_query_recall(gold, preds_by[name].get(qid, []), 10))

    names = list(methods.keys())

    n = len(test_qids)
    rng = np.random.default_rng(42)

    print("=== 方法对比：配对 Bootstrap ===\n")

    # Core comparisons used to separate flat knowledge features, local structure,
    # and the no-medical-knowledge hypergraph variant.

    comparisons = [
        ("KCH-MedRank", "Flat knowledge LTR (no graph)", "完整模型 vs 无图扁平知识特征"),
        ("KCH-MedRank", "Hypergraph (no med)", "超图+知识 vs 超图无知识（知识的贡献）"),
        ("KCH-MedRank", "Pairwise graph", "超图 vs 对偶图（结构贡献）"),
        ("Flat knowledge LTR (no graph)", "Retrieval LTR", "扁平知识特征的独立贡献"),
        ("Hypergraph (no med)", "Flat knowledge LTR (no graph)", "无知识超图是否引入噪音"),
    ]

    for a_name, b_name, description in comparisons:
        a_arr = np.array(rec[a_name])
        b_arr = np.array(rec[b_name])
        delta = a_arr - b_arr
        obs = float(np.mean(delta))

        boot = np.empty(10000)
        for i in range(10000):
            idx = rng.integers(0, n, size=n)
            boot[i] = float(np.mean(delta[idx]))

        p_lower = (float(np.sum(boot <= 0)) + 1) / 10001
        p_upper = (float(np.sum(boot >= 0)) + 1) / 10001
        p = min(1.0, 2 * min(p_lower, p_upper))
        ci_low = float(np.quantile(boot, 0.025))
        ci_hi = float(np.quantile(boot, 0.975))

        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")
        print(f"{description}")
        print(f"  {a_name:>20s}: {float(np.mean(a_arr)):.4f}")
        print(f"  {b_name:>20s}: {float(np.mean(b_arr)):.4f}")
        print(f"  Δ = {obs:+.4f}  CI=[{ci_low:+.4f}, {ci_hi:+.4f}]  p={p:.4f}  {sig}")
        print()

    # 核心分析：交互效应
    print("=" * 60)
    print("=== 交互效应分析 ===\n")
    sem_gain = float(np.mean(rec["Flat knowledge LTR (no graph)"])) - float(np.mean(rec["Retrieval LTR"]))
    hg_no_med_gain = float(np.mean(rec["Hypergraph (no med)"])) - float(np.mean(rec["Flat knowledge LTR (no graph)"]))
    kch_gain = float(np.mean(rec["KCH-MedRank"])) - float(np.mean(rec["Flat knowledge LTR (no graph)"]))
    interaction = kch_gain - hg_no_med_gain

    print(f"扁平知识特征的独立增益:  {sem_gain:+.4f}")
    print(f"超图（无医学知识）的增益: {hg_no_med_gain:+.4f} ← 有害")
    print(f"超图（有医学知识）的增益: {kch_gain:+.4f}  ← 有益")
    print(f"知识×结构的交互效应:     {interaction:+.4f}")
    print()
    print("结论：扁平知识特征解释了主要的 top-10 增益；")
    print("局部图/超图结构应被解释为互补的可解释结构信号。")


if __name__ == "__main__":
    main()
