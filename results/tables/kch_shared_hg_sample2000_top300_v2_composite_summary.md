# KCH-MedRank sample2000 v2 diagnostic summary

Date: 2026-05-30

Scope: diagnostic BioASQ sample2000, numeric qid order, modulo split with 1,200 train / 400 validation / 400 test queries. This is not a full-test result.

Candidate pool:
- Base: `outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl`
- Expansion: shared-candidate concept clusters appended after preserved base top100
- Output: `outputs/retrieval/concept_hg_shared_clusters_sample2000_top300_v2.jsonl`
- Candidate coverage: Recall@100 = 0.7831, Recall@200 = 0.8921, Recall@300 = 0.9104

Method run:
- Script: `scripts/run_kch_medrank.py`
- Output prefix: `kch_shared_hg_sample2000_top300_v2_composite`
- Validation objective: `composite = recall@10 + 0.5 * ndcg@10 + 0.2 * mrr@10`

Primary test results:

| Method | Recall@10 | MRR@10 | nDCG@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| Hybrid RRF | 0.4485 | 0.8509 | 0.6337 | 0.7835 |
| Flat biomedical knowledge LTR | 0.5173 | 0.8719 | 0.6958 | 0.8813 |
| Structural hypergraph LTR | 0.5075 | 0.8656 | 0.6833 | 0.8783 |
| Pairwise graph LTR | 0.5180 | 0.8675 | 0.6962 | 0.8834 |
| Full KCH-MedRank | 0.5136 | 0.8562 | 0.6885 | 0.8821 |
| Remove diffusion/centrality | 0.5116 | 0.8612 | 0.6905 | 0.8818 |

Paired bootstrap, 5,000 resamples:

| Comparison | Metric@10 | Delta | 95% CI | p |
| --- | --- | ---: | ---: | ---: |
| Full vs Hybrid | Recall | +0.0651 | [+0.0511, +0.0803] | 0.0004 |
| Full vs Hybrid | MRR | +0.0053 | [-0.0126, +0.0230] | 0.5643 |
| Full vs Hybrid | nDCG | +0.0548 | [+0.0429, +0.0670] | 0.0004 |
| Full vs Flat knowledge LTR | MRR | -0.0157 | [-0.0268, -0.0052] | 0.0032 |
| Full vs Flat knowledge LTR | nDCG | -0.0073 | [-0.0121, -0.0025] | 0.0028 |
| Full vs Pairwise graph LTR | MRR | -0.0114 | [-0.0213, -0.0017] | 0.0224 |
| Full vs Pairwise graph LTR | nDCG | -0.0077 | [-0.0127, -0.0029] | 0.0008 |
| Full vs Remove diffusion/centrality | Recall | +0.0020 | [-0.0037, +0.0084] | 0.4939 |
| Full vs Structural hypergraph LTR | nDCG | +0.0052 | [-0.0025, +0.0129] | 0.1844 |

Interpretation boundary:
- Stable support: shared-cluster expansion plus supervised KCH-style reranking strongly improves top-10 evidence coverage and nDCG over Hybrid RRF.
- Not supported on sample2000: Full KCH-MedRank is not stronger than flat biomedical knowledge LTR or pairwise graph LTR. Pairwise/flat variants are better on nDCG@10 and MRR@10 in this diagnostic.
- Hypergraph-specific conclusion should remain cautious: structural and source-reason features are interpretable complementary signals, but this run does not prove that n-ary hypergraph structure is the dominant top-k driver.
- Hard subset remains diagnostic only: test hard subset has 14 queries, validation hard subset has 26 queries.
