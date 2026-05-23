"""Task 3 (v3): 超参数敏感性分析 — 纯扩散层面分析。

策略：只重跑 diffusion() 而非重建整个超图。对样本问题构建一次超图，
然后用不同 (damping, iterations) 组合跑扩散，比较 passage score 的相关性和稳定性。
直接从 v2 实验的完整特征中提取 hypergraph_score，分析其在不同扩散参数下的值分布。
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import group_predictions
from src.hypergraph.local import build_local_hypergraph, diffuse, weighted_degree_centrality
from src.rerank.hypergraph import entity_map, mesh_map
from src.knowledge.mesh_hierarchy import load_mesh_hierarchy
from src.utils import read_jsonl

SAMPLE_SIZE = 30


def main():
    base = PROJECT_ROOT

    print("Loading predictions...")
    hybrid_preds = read_jsonl(str(base / "outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl"))

    q_entities = entity_map(read_jsonl(str(base / "data/processed/bioasq_question_entities.jsonl")), "question_id")
    p_entities = entity_map(read_jsonl(str(base / "data/processed/bioasq_passage_entities.jsonl")), "passage_id")
    q_mesh = mesh_map(read_jsonl(str(base / "data/processed/bioasq_question_mesh.jsonl")), "question_id")
    p_mesh = mesh_map(read_jsonl(str(base / "data/processed/bioasq_passage_mesh.jsonl")), "passage_id")
    mesh_hierarchy = load_mesh_hierarchy(read_jsonl(str(base / "data/external_knowledge/mesh_hierarchy_2026.jsonl")))
    entity_relations: dict = {}  # skip PrimeKG for speed, relations don't affect diffusion analysis

    preds_by_qid = group_predictions(hybrid_preds)
    test_qids = sorted([qid for qid in preds_by_qid if int(qid) % 5 == 4])[:SAMPLE_SIZE]
    print(f"Sample questions: {len(test_qids)}")

    damping_grid = [0.7, 0.8, 0.85, 0.9, 0.95]
    iterations_grid = [3, 5, 7, 10]

    all_correlations: dict[str, list[float]] = defaultdict(list)
    all_diffs: dict[str, list[float]] = defaultdict(list)

    for qid in test_qids:
        candidates = preds_by_qid[qid][:100]
        for row in candidates:
            row["passage_id"] = str(row["passage_id"])

        graph = build_local_hypergraph(
            qid, candidates,
            q_entities.get(qid, []),
            {str(pid): ents for pid, ents in p_entities.items()
             if str(pid) in {str(r["passage_id"]) for r in candidates}},
            question_mesh=q_mesh.get(qid, []),
            passage_mesh=p_mesh,
            mesh_hierarchy=mesh_hierarchy,
            entity_relations=entity_relations,
            structure="knowledge_hypergraph",
        )

        baseline_scores = diffuse(graph, [graph.question_node], iterations=5, damping=0.85)

        for damping in damping_grid:
            for iterations in iterations_grid:
                scores = diffuse(graph, [graph.question_node], iterations=iterations, damping=damping)
                base_vals = np.array([baseline_scores.get(p_node, 0.0) for p_node in graph.passage_nodes.values()])
                test_vals = np.array([scores.get(p_node, 0.0) for p_node in graph.passage_nodes.values()])

                base_nonzero = base_vals > 0
                if base_nonzero.sum() < 2:
                    continue

                corr = float(np.corrcoef(base_vals[base_nonzero], test_vals[base_nonzero])[0, 1])
                diff = float(np.abs(base_vals - test_vals).mean())

                key = f"d={damping}_t={iterations}"
                all_correlations[key].append(corr if not np.isnan(corr) else 0.0)
                all_diffs[key].append(diff)

    lines = []
    lines.append("# Hyperparameter Sensitivity: Diffusion Stability Analysis")
    lines.append("")
    lines.append(f"Analysis on {SAMPLE_SIZE} test questions. For each question, the hypergraph is built once.")
    lines.append(f"Diffusion is run with varied (damping, iterations). Passage scores from baseline (d=0.85, t=5)")
    lines.append(f"are correlated with scores from each parameter combination.")
    lines.append("")
    lines.append("| damping | iterations | Pearson r (mean) | r (std) | |Δscore| (mean) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")

    for damping in damping_grid:
        for iterations in iterations_grid:
            key = f"d={damping}_t={iterations}"
            corrs = all_correlations[key]
            diffs = all_diffs[key]
            if corrs:
                r_mean = float(np.mean(corrs))
                r_std = float(np.std(corrs))
                d_mean = float(np.mean(diffs))
                lines.append(f"| {damping} | {iterations} | {r_mean:.4f} | {r_std:.4f} | {d_mean:.6f} |")

    lines.append("")
    lines.append("*Pearson r measures linear correlation of per-passage diffusion scores between the baseline (d=0.85, t=5) and the alternative parameters.*")
    lines.append("*|Δscore| is the mean absolute per-passage score difference.*")
    lines.append("")
    lines.append("*Interpretation: r > 0.95 indicates near-identical passage ordering; r > 0.85 suggests very similar ordering.*")

    out_md = base / "results/tables/hyperparameter_sensitivity.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {out_md}")
    print("Done.")


if __name__ == "__main__":
    main()
