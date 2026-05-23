"""Task 1: 可解释性量化分析脚本。

从 raw 预测数据重建 rescued/lost 案例，分类三种超图机制：
1. MeSH 层级路径（Mesh Hierarchy Path）
2. 共享实体簇路径（Shared Entity Cluster Path）  
3. PrimeKG 关系路径（Relation Path）

产出：results/tables/interpretability_mechanisms.md + 路径追踪统计。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import group_predictions, group_qrels
from src.utils import read_jsonl, write_json

TOP_K = 10


def _question_type_keywords(question: str) -> str:
    q = question.lower()
    patterns = [
        ("yes/no", ["is ", "are ", "do ", "does ", "can ", "could ", "will ",
                    "is there", "are there", "has ", "have "]),
        ("list/synthesis", ["list ", "what are ", "which are "]),
        ("what_is", ["what is"]),
        ("which", ["which "]),
        ("how", ["how "]),
    ]
    for category, keywords in patterns:
        for kw in keywords:
            if q.startswith(kw) or f" {kw}" in q:
                return category

    mechanism_kw = ["mechanism", "pathway", "signaling", "involved in"]
    for kw in mechanism_kw:
        if kw in q:
            return "mechanism"
    treatment_kw = ["treatment", "therapy", "drug", "treated", "treat", "therapeutic"]
    for kw in treatment_kw:
        if kw in q:
            return "treatment"
    gene_kw = ["gene", "mutation", "genetic", "genomic", "protein", "receptor", "molecular"]
    for kw in gene_kw:
        if kw in q:
            return "molecular"
    return "other"


def load_jsonl_features(jsonl_path: str) -> dict[str, dict[str, dict]]:
    if not Path(jsonl_path).exists():
        return {}
    preds = read_jsonl(jsonl_path)
    feats: dict[str, dict[str, dict]] = {}
    for row in preds:
        qid = str(row.get("question_id", ""))
        pid = str(row.get("passage_id", ""))
        meta = row.get("metadata", {})
        feat = meta.get("features", {})
        rank = row.get("rank", 100)
        feats.setdefault(qid, {})
        feats[qid][pid] = {"rank": rank, "features": feat}
    return feats


def compute_rescued_lost(
    qrels_by_qid: dict[str, set[str]],
    hybrid_by_qid: dict[str, list[dict]],
    kch_by_qid: dict[str, list[dict]],
) -> tuple[list[dict], list[dict]]:
    rescued_cases = []
    lost_cases = []

    test_qids = {qid for qid in hybrid_by_qid if int(qid) % 5 == 4}

    for qid in sorted(test_qids):
        gold_ids = qrels_by_qid.get(qid, set())
        if not gold_ids:
            continue
        hybrid_rows = hybrid_by_qid.get(qid, [])
        kch_rows = kch_by_qid.get(qid, [])
        if not hybrid_rows or not kch_rows:
            continue

        hybrid_ranks: dict[str, int] = {}
        for r in hybrid_rows:
            pid = str(r.get("passage_id", ""))
            hybrid_ranks[pid] = r.get("rank", 100)

        kch_ranks: dict[str, int] = {}
        for r in kch_rows:
            pid = str(r.get("passage_id", ""))
            kch_ranks[pid] = r.get("rank", 100)

        for pid in gold_ids:
            h_rank = hybrid_ranks.get(pid, 100)
            k_rank = kch_ranks.get(pid, 100)
            rank_delta = h_rank - k_rank

            if h_rank > TOP_K and k_rank <= TOP_K:
                rescued_cases.append({
                    "question_id": qid,
                    "passage_id": pid,
                    "baseline_rank": h_rank,
                    "candidate_rank": k_rank,
                    "rank_gain": rank_delta,
                })
            elif h_rank <= TOP_K and k_rank > TOP_K:
                lost_cases.append({
                    "question_id": qid,
                    "passage_id": pid,
                    "baseline_rank": h_rank,
                    "candidate_rank": k_rank,
                    "rank_loss": k_rank - h_rank,
                })

    return rescued_cases, lost_cases


def classify_mechanism(features: dict) -> dict:
    mesh_ov_count = float(features.get("mesh_overlap_count", 0))
    entity_ov_count = float(features.get("entity_overlap_count", 0))
    entity_cov = float(features.get("question_entity_coverage", 0))
    relation_count = float(features.get("primekg_relation_count", 0))
    shared_entity_edges = float(features.get("local_shared_entity_edges", 0))
    shared_mesh_parent = float(features.get("shared_mesh_parent_cluster_size", 0))
    mesh_hierarchy_edges = float(features.get("local_mesh_hierarchy_edges", 0))
    mesh_ancestor_edges = float(features.get("local_mesh_ancestor_edges", 0))
    relation_edges = float(features.get("local_primekg_relation_edges", 0))
    mesh_tree_sim = float(features.get("mesh_tree_similarity_mean", 0))
    mesh_tree_dist = float(features.get("mesh_tree_distance_min", 0))
    shared_mesh_term = float(features.get("shared_mesh_term_cluster_size", 0))

    mesh_hier_score = 0.0
    if mesh_ov_count > 0:
        mesh_hier_score += 1.0
    if shared_mesh_parent > 0:
        mesh_hier_score += min(shared_mesh_parent / 50.0, 1.0)
    if shared_mesh_term > 0:
        mesh_hier_score += min(shared_mesh_term / 50.0, 0.3)
    if mesh_ancestor_edges > 0:
        mesh_hier_score += min(mesh_ancestor_edges / 50.0, 0.5)
    if mesh_tree_sim > 0:
        mesh_hier_score += mesh_tree_sim * 0.5
    if mesh_tree_dist > 0 and mesh_tree_dist < 10:
        mesh_hier_score += (10.0 - mesh_tree_dist) / 10.0

    entity_cluster_score = 0.0
    if entity_ov_count > 0:
        entity_cluster_score += min(entity_ov_count / 3.0, 1.0)
    if entity_cov > 0:
        entity_cluster_score += entity_cov
    if shared_entity_edges > 0:
        entity_cluster_score += min(shared_entity_edges / 128.0, 1.0)

    relation_score = 0.0
    if relation_count > 0:
        relation_score += min(relation_count / 3.0, 1.0)
    if relation_edges > 0:
        relation_score += min(relation_edges / 50.0, 1.0)

    scores = {
        "mesh_hierarchy": mesh_hier_score,
        "entity_cluster": entity_cluster_score,
        "relation_path": relation_score,
    }
    primary = max(scores, key=scores.get)
    max_score = scores[primary]
    if max_score <= 0.15:
        primary = "diffusion_only"

    return {
        "primary_mechanism": primary,
        "scores": scores,
        "mesh_overlap_count": mesh_ov_count,
        "entity_overlap_count": entity_ov_count,
        "question_entity_coverage": entity_cov,
        "relation_count": relation_count,
        "shared_entity_edges": shared_entity_edges,
        "shared_mesh_parent_size": shared_mesh_parent,
        "mesh_hierarchy_edges": mesh_hierarchy_edges,
        "relation_edges": relation_edges,
    }


def load_question_texts(questions_path: str) -> dict[str, str]:
    if not Path(questions_path).exists():
        return {}
    rows = read_jsonl(questions_path)
    return {str(r["question_id"]): r.get("question", "") for r in rows}


def _safe_get(mapping: dict, key: str, default: dict) -> dict:
    result = mapping.get(key, default)
    if "cases" not in result:
        result["cases"] = []
    if "rank_gains" not in result:
        result["rank_gains"] = []
    if "count" not in result:
        result["count"] = 0
    return result


DEFAULT_MECH_STATS = {"count": 0, "cases": [], "rank_gains": []}


def main():
    base = PROJECT_ROOT

    kch_pred_path = base / "outputs" / "rerank" / "kch_medrank_enhanced_bioasq_v2_full_kch_medrank_test_top100.jsonl"
    hybrid_pred_path = base / "outputs" / "retrieval" / "enhanced_hybrid_w122_full_top100.jsonl"
    qrels_path = base / "data" / "processed" / "bioasq_qrels.jsonl"
    questions_path = base / "data" / "processed" / "bioasq_questions.jsonl"
    output_json = base / "results" / "metrics" / "interpretability_analysis.json"
    output_md = base / "results" / "tables" / "interpretability_mechanisms.md"

    print("Loading qrels...")
    qrels_by_qid = group_qrels(read_jsonl(str(qrels_path)))
    qrels_loaded = {str(k): set(str(p) for p in v) for k, v in qrels_by_qid.items()}
    print(f"  Loaded qrels for {len(qrels_loaded)} questions")

    print("Loading KCH-MedRank predictions...")
    kch_preds = read_jsonl(str(kch_pred_path))
    kch_by_qid = group_predictions(kch_preds)
    kch_feats = load_jsonl_features(str(kch_pred_path))
    print(f"  Loaded features for {len(kch_feats)} questions")

    print("Loading Hybrid predictions...")
    hybrid_preds = read_jsonl(str(hybrid_pred_path))
    hybrid_by_qid = group_predictions(hybrid_preds)
    print(f"  Loaded hybrid for {len(hybrid_by_qid)} questions")

    print("Loading question texts...")
    q_texts = load_question_texts(str(questions_path))

    print("Computing rescued/lost cases...")
    rescued, lost = compute_rescued_lost(qrels_loaded, hybrid_by_qid, kch_by_qid)
    print(f"  Rescued (outside→inside top-{TOP_K}): {len(rescued)} cases")
    print(f"  Lost (inside→outside top-{TOP_K}): {len(lost)} cases")

    rescued_details = []
    lost_details = []

    mechanism_stats_rescued: dict[str, dict] = {}
    mechanism_stats_lost: dict[str, dict] = {}

    def _ensure_mech(mapping: dict, key: str) -> dict:
        if key not in mapping:
            mapping[key] = {"count": 0, "cases": [], "rank_gains": []}
        return mapping[key]

    for case in rescued:
        qid = str(case["question_id"])
        pid = str(case["passage_id"])
        baseline_rank = case["baseline_rank"]
        candidate_rank = case["candidate_rank"]
        rank_gain = case["rank_gain"]
        question = q_texts.get(qid, "")
        q_type = _question_type_keywords(question)

        kch_feat = kch_feats.get(qid, {}).get(pid, {})
        features = kch_feat.get("features", {})

        if not features:
            continue

        mech_result = classify_mechanism(features)
        primary = mech_result["primary_mechanism"]

        detail = {
            "question_id": qid,
            "passage_id": pid,
            "question": question[:120],
            "question_type": q_type,
            "baseline_rank": baseline_rank,
            "candidate_rank": candidate_rank,
            "rank_gain": rank_gain,
            **mech_result,
        }
        rescued_details.append(detail)

        m = _ensure_mech(mechanism_stats_rescued, primary)
        m["count"] += 1
        m["cases"].append({"qid": qid, "pid": pid, "gain": rank_gain, "q_type": q_type})
        m["rank_gains"].append(rank_gain)

    for case in lost:
        qid = str(case["question_id"])
        pid = str(case["passage_id"])
        baseline_rank = case["baseline_rank"]
        candidate_rank = case["candidate_rank"]
        rank_loss = case["rank_loss"]
        question = q_texts.get(qid, "")
        q_type = _question_type_keywords(question)

        kch_feat = kch_feats.get(qid, {}).get(pid, {})
        features = kch_feat.get("features", {})

        if not features:
            continue

        mech_result = classify_mechanism(features)
        primary = mech_result["primary_mechanism"]

        detail = {
            "question_id": qid,
            "passage_id": pid,
            "question": question[:120],
            "question_type": q_type,
            "baseline_rank": baseline_rank,
            "candidate_rank": candidate_rank,
            "rank_loss": rank_loss,
            **mech_result,
        }
        lost_details.append(detail)

        m = _ensure_mech(mechanism_stats_lost, primary)
        m["count"] += 1
        m["cases"].append({"qid": qid, "pid": pid, "loss": rank_loss, "q_type": q_type})
        m["rank_gains"].append(-rank_loss)

    stat_keys = ["mesh_hierarchy", "entity_cluster", "relation_path", "diffusion_only"]

    print("\nGenerating report...")
    lines = []
    lines.append("# Interpretability Analysis: Three-Mechanism Classification")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Rescued (outside→inside top-{TOP_K}): {len(rescued_details)} gold passages")
    lines.append(f"- Lost (inside→outside top-{TOP_K}): {len(lost_details)} gold passages")
    lines.append("")
    lines.append("### Three Interpretable Mechanisms")
    lines.append("1. **MeSH Hierarchy Path**: Question and passage share MeSH concepts through hierarchical tree paths")
    lines.append("2. **Shared Entity Cluster Path**: Gold passage and other candidates share biomedical entities, forming hyperedge clusters")
    lines.append("3. **PrimeKG Relation Path**: Question entities are relationally linked to passage entities via PrimeKG")
    lines.append("4. **Diffusion Only**: No explicit knowledge alignment; rescue from hypergraph topology and diffusion propagation alone")
    lines.append("")

    lines.append("## Rescued Cases: Mechanism Distribution")
    lines.append(f"Total rescued cases analyzed: {len(rescued_details)}")
    lines.append("")
    lines.append("| Mechanism | Count | % | Avg Rank Gain | Q-Type Breakdown |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for key in stat_keys:
        m = _safe_get(mechanism_stats_rescued, key, DEFAULT_MECH_STATS.copy())
        count = m["count"]
        pct = count / len(rescued_details) * 100 if rescued_details else 0
        avg_gain = sum(m["rank_gains"]) / len(m["rank_gains"]) if m["rank_gains"] else 0
        q_types = Counter(c["q_type"] for c in m["cases"])
        q_type_str = ", ".join(f"{t}: {c}" for t, c in q_types.most_common(4))
        lines.append(f"| {key} | {count} | {pct:.1f} | {avg_gain:.1f} | {q_type_str} |")

    lines.append("")
    lines.append("## Lost Cases: Mechanism Distribution")
    lines.append(f"Total lost cases analyzed: {len(lost_details)}")
    lines.append("")
    lines.append("| Mechanism | Count | % | Avg Rank Loss | Q-Type Breakdown |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for key in stat_keys:
        m = _safe_get(mechanism_stats_lost, key, DEFAULT_MECH_STATS.copy())
        count = m["count"]
        pct = count / len(lost_details) * 100 if lost_details else 0
        avg_loss = sum(abs(v) for v in m["rank_gains"]) / len(m["rank_gains"]) if m["rank_gains"] else 0
        q_types = Counter(c["q_type"] for c in m["cases"])
        q_type_str = ", ".join(f"{t}: {c}" for t, c in q_types.most_common(4))
        lines.append(f"| {key} | {count} | {pct:.1f} | {avg_loss:.1f} | {q_type_str} |")

    lines.append("")
    lines.append("## Mechanism Activation Statistics (Rescued Cases)")
    lines.append("")
    lines.append("| Metric | MeSH Hierarchy | Entity Cluster | Relation Path | Diffusion Only |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")

    for metric_key in ["mesh_overlap_count", "entity_overlap_count", "question_entity_coverage",
                        "relation_count", "shared_entity_edges", "shared_mesh_parent_size"]:
        vals: dict[str, list[float]] = defaultdict(list)
        for d in rescued_details:
            vals[d["primary_mechanism"]].append(d.get(metric_key, 0))
        parts = [metric_key]
        for key in stat_keys:
            vlist = vals.get(key, [])
            avg = sum(vlist) / len(vlist) if vlist else 0
            parts.append(f"{avg:.2f}")
        lines.append(f"| {' | '.join(parts)} |")

    lines.append("")
    lines.append("## Path Tracing: Top Rescued Cases by Mechanism")
    lines.append("")

    for key in stat_keys:
        m = _safe_get(mechanism_stats_rescued, key, DEFAULT_MECH_STATS.copy())
        cases_list = m["cases"]
        cases_sorted = sorted(cases_list, key=lambda c: c["gain"], reverse=True)[:5]
        if cases_sorted:
            lines.append(f"### Top {key} Cases (largest rank gains)")
            lines.append("")
            lines.append("| Q-ID | P-ID | Rank Gain | Q-Type | Question |")
            lines.append("| --- | --- | ---: | --- | --- |")
            for c in cases_sorted:
                qtext = q_texts.get(c["qid"], "")[:100]
                lines.append(f"| {c['qid']} | {c['pid']} | {c['gain']} | {c['q_type']} | {qtext} |")
            lines.append("")

    lines.append("## Path Tracing: Top Lost Cases by Mechanism")
    lines.append("")

    for key in stat_keys:
        m = _safe_get(mechanism_stats_lost, key, DEFAULT_MECH_STATS.copy())
        cases_list = m["cases"]
        cases_sorted = sorted(cases_list, key=lambda c: c.get("loss", 0), reverse=True)[:5]
        if cases_sorted:
            lines.append(f"### Top {key} Lost Cases (largest rank losses)")
            lines.append("")
            lines.append("| Q-ID | P-ID | Rank Loss | Q-Type | Question |")
            lines.append("| --- | --- | ---: | --- | --- |")
            for c in cases_sorted:
                qtext = q_texts.get(c["qid"], "")[:100]
                lines.append(f"| {c['qid']} | {c['pid']} | {c.get('loss', c.get('loss', 0))} | {c['q_type']} | {qtext} |")
            lines.append("")

    lines.append("")
    lines.append("## Mechanism Co-occurrence Analysis (Rescued Cases)")
    lines.append("")
    lines.append("Cases where multiple mechanisms simultaneously contributed to rescue:")
    lines.append("")

    multi_mech: dict[str, int] = {}
    for d in rescued_details:
        active = []
        scores = d.get("scores", {})
        for mkey in ["mesh_hierarchy", "entity_cluster", "relation_path"]:
            if scores.get(mkey, 0) > 0.3:
                active.append(mkey)
        combo = " + ".join(active) if active else "diffusion_only"
        multi_mech[combo] = multi_mech.get(combo, 0) + 1

    lines.append("| Active Mechanisms | Count | % |")
    lines.append("| --- | ---: | ---: |")
    total_q = len(rescued_details)
    for combo, count in sorted(multi_mech.items(), key=lambda x: -x[1]):
        lines.append(f"| {combo} | {count} | {count/total_q*100:.1f} |")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {output_md}")

    output_data = {
        "n_rescued": len(rescued_details),
        "n_lost": len(lost_details),
        "rescued_summary": {k: {"count": v["count"], "avg_rank_gain": sum(v["rank_gains"])/len(v["rank_gains"]) if v["rank_gains"] else 0}
                            for k, v in mechanism_stats_rescued.items()},
        "lost_summary": {k: {"count": v["count"], "avg_rank_loss": sum(abs(x) for x in v["rank_gains"])/len(v["rank_gains"]) if v["rank_gains"] else 0}
                         for k, v in mechanism_stats_lost.items()},
        "rescued_details": rescued_details,
        "lost_details": lost_details,
        "multi_mechanism_cooccurrence": multi_mech,
    }
    write_json(str(output_json), output_data)
    print(f"JSON output written to {output_json}")


if __name__ == "__main__":
    main()
