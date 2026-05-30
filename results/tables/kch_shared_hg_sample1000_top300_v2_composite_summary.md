# KCH-MedRank sample1000 v2 diagnostic summary

Date: 2026-05-30

Scope: diagnostic BioASQ sample1000, numeric qid order, modulo split with 600 train / 200 validation / 200 test queries. This is not a full-test result.

Candidate pool:
- Base: `outputs/retrieval/enhanced_hybrid_w122_full_top100.jsonl`
- Expansion: shared-candidate concept clusters appended after preserved base top100
- Output: `outputs/retrieval/concept_hg_shared_clusters_sample1000_top300_v2.jsonl`
- Candidate coverage: Recall@100 = 0.7841, Recall@200 = 0.8923, Recall@300 = 0.9097

Method run:
- Script: `scripts/run_kch_medrank.py`
- Output prefix: `kch_shared_hg_sample1000_top300_v2_composite`
- Validation objective: `composite = recall@10 + 0.5 * ndcg@10 + 0.2 * mrr@10`
- Key feature changes: source-reason aggregates, activation features, noise penalties, base metadata preservation, MedCPT/dense metadata fallback, and actual composite/constrained/hard-weighted selection support.

Primary test results:

| Method | Recall@10 | MRR@10 | nDCG@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| Hybrid RRF | 0.4388 | 0.8493 | 0.6095 | 0.7729 |
| Flat biomedical knowledge LTR | 0.5101 | 0.8786 | 0.6809 | 0.8861 |
| Structural hypergraph LTR | 0.5036 | 0.8673 | 0.6642 | 0.8801 |
| Pairwise graph LTR | 0.5097 | 0.8659 | 0.6767 | 0.8849 |
| Full KCH-MedRank | 0.5066 | 0.8841 | 0.6762 | 0.8860 |
| Remove diffusion/centrality | 0.5038 | 0.8670 | 0.6735 | 0.8858 |

Paired bootstrap, 5,000 resamples:

| Comparison | Metric@10 | Delta | 95% CI | p |
| --- | --- | ---: | ---: | ---: |
| Full vs Hybrid | Recall | +0.0678 | [+0.0475, +0.0887] | 0.0004 |
| Full vs Hybrid | MRR | +0.0348 | [+0.0029, +0.0662] | 0.0284 |
| Full vs Hybrid | nDCG | +0.0667 | [+0.0498, +0.0847] | 0.0004 |
| Full vs Flat knowledge LTR | Recall | -0.0036 | [-0.0128, +0.0066] | 0.4543 |
| Full vs Flat knowledge LTR | MRR | +0.0055 | [-0.0049, +0.0167] | 0.2891 |
| Full vs Pairwise graph LTR | MRR | +0.0182 | [+0.0037, +0.0350] | 0.0124 |
| Full vs Pairwise graph LTR | nDCG | -0.0005 | [-0.0076, +0.0055] | 0.8746 |
| Full vs Remove diffusion/centrality | MRR | +0.0171 | [+0.0026, +0.0336] | 0.0200 |
| Full vs Remove diffusion/centrality | Recall | +0.0027 | [-0.0068, +0.0120] | 0.5627 |
| Full vs Structural hypergraph LTR | nDCG | +0.0120 | [+0.0012, +0.0231] | 0.0300 |

Interpretation boundary:
- Supported: this diagnostic sample shows substantial top-10 gains over the Hybrid RRF baseline, including Recall@10, MRR@10, and nDCG@10.
- Partially supported: structural hypergraph features improve some rank-quality signals over pairwise/remove-diffusion variants, especially MRR or nDCG, but not Recall@10.
- Not supported yet: Full KCH-MedRank does not significantly beat flat biomedical knowledge LTR on top-10 metrics in this sample.
- Hard subset remains diagnostic only: test hard subset has 7 queries, validation hard subset has 12 queries.
