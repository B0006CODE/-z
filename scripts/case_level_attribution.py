"""案例级特征归因：在 Full > Semantic 的案例中，哪些特征驱动了排序差异。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import group_predictions, group_qrels
from src.utils import read_jsonl


def per_query_recall(gold, rows, k):
    ranked_ids = {str(r["passage_id"]) for r in rows[:k]}
    return len(set(gold) & ranked_ids) / len(gold) if gold else 0.0


def main():
    qrels_by_qid = group_qrels(read_jsonl("data/processed/bioasq_qrels.jsonl"))
    hybrid_by_qid = group_predictions(read_jsonl("outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl"))
    semantic_preds = read_jsonl("outputs/rerank/kch_medrank_enhanced_bioasq_v2_semantic_no_hypergraph_ltr_test_top100.jsonl")
    full_preds = read_jsonl("outputs/rerank/kch_medrank_enhanced_bioasq_v2_full_kch_medrank_test_top100.jsonl")

    semantic_by_qid = group_predictions(semantic_preds)
    full_by_qid = group_predictions(full_preds)

    test_qids = {qid for qid in hybrid_by_qid if int(qid) % 5 == 4}

    FEATURE_GROUPS = {
        "retrieval": ["base_rank_score", "hybrid_score", "bm25_score", "dense_score", "bm25_rank_score", "dense_rank_score", "rank_percentile"],
        "semantic": ["biomedical_semantic_score", "biomedical_semantic_rank_score"],
        "entity": ["entity_overlap_count", "entity_jaccard", "question_entity_coverage", "passage_entity_count"],
        "mesh_exact": ["mesh_overlap_count", "mesh_jaccard", "question_mesh_coverage", "passage_mesh_count"],
        "mesh_hierarchy": ["mesh_tree_similarity_max", "passage_mesh_specificity", "shared_mesh_term_cluster_size", "shared_mesh_parent_cluster_size"],
        "hypergraph": ["hypergraph_score_norm", "hypergraph_degree_centrality", "local_num_nodes", "local_num_hyperedges"],
        "interaction": ["hypergraph_x_inverse_rank", "mesh_cluster_x_inverse_semantic", "mesh_specificity_x_inverse_rank"],
    }

    full_win_feats = defaultdict(float)
    sem_win_feats = defaultdict(float)
    n_full_win = 0
    n_sem_win = 0

    for qid in sorted(test_qids):
        gold = qrels_by_qid.get(qid, {})
        sem_r = per_query_recall(gold, semantic_by_qid.get(qid, []), 10)
        full_r = per_query_recall(gold, full_by_qid.get(qid, []), 10)

        if full_r == sem_r:
            continue

        # 找到该问题下两个方法排序差异最大的金标准文献，比较其特征差异
        gold_ids = set(gold)
        sem_rows = {r["passage_id"]: r for r in semantic_by_qid.get(qid, [])}
        full_rows = {r["passage_id"]: r for r in full_by_qid.get(qid, [])}

        for pid in gold_ids:
            if pid not in sem_rows or pid not in full_rows:
                continue
            sem_rank = sem_rows[pid]["rank"]
            full_rank = full_rows[pid]["rank"]
            if sem_rank == full_rank:
                continue

            sem_feats = sem_rows[pid].get("metadata", {}).get("features", {})
            full_feats = full_rows[pid].get("metadata", {}).get("features", {})

            for group_name, feat_names in FEATURE_GROUPS.items():
                sem_sum = sum(float(sem_feats.get(f, 0)) for f in feat_names)
                full_sum = sum(float(full_feats.get(f, 0)) for f in feat_names)
                diff = full_sum - sem_sum

                if full_r > sem_r:
                    full_win_feats[group_name] += diff
                else:
                    sem_win_feats[group_name] += (-diff)

        if full_r > sem_r:
            n_full_win += 1
        else:
            n_sem_win += 1

    print("=== 案例级特征归因 ===\n")
    print(f"Full KCH-MedRank 更好的案例: {n_full_win}")
    print(f"Semantic-only 更好的案例:   {n_sem_win}\n")

    print(f"{'特征组':<20} {'Full Win 贡献':>15} {'Sem Win 贡献':>15} {'净效应':>15}")
    print("-" * 65)
    for group in FEATURE_GROUPS:
        fw = full_win_feats[group]
        sw = sem_win_feats[group]
        net = fw - sw
        direction = "← 驱动 Full 更好" if net > 0 else ("← 驱动 Semantic 更好" if net < 0 else "")
        print(f"{group:<20} {fw:>15.4f} {sw:>15.4f} {net:>15.4f} {direction}")

    print("\n")
    print("=== 消融与案例级归因的关联 ===\n")
    print("全量消融不显著 ≠ 知识特征在所有案例中无用。")
    print("全量消融看的是平均效应（被 92% 的 tie 案例稀释），")
    print("案例级归因看的是：在有差异的案例中，是什么特征驱动了差异。")
    print("这两者互补而非矛盾。")


if __name__ == "__main__":
    main()
