| Dataset | Comparison | Metric | k | Baseline | Candidate | Delta | 95% CI | Relative | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_enhanced_v2 | Full KCH-MedRank vs Flat knowledge LTR, no graph | MRR | 10 | 0.7872 | 0.7867 | -0.0004 | [-0.0056, +0.0047] | -0.05% | 0.8635 |
| bioasq_enhanced_v2 | Full KCH-MedRank vs Flat knowledge LTR, no graph | Recall | 10 | 0.5309 | 0.5329 | +0.0020 | [-0.0011, +0.0058] | +0.37% | 0.2540 |
| bioasq_enhanced_v2 | Full KCH-MedRank vs Flat knowledge LTR, no graph | Ndcg | 10 | 0.6436 | 0.6433 | -0.0003 | [-0.0028, +0.0023] | -0.04% | 0.8157 |
| bioasq_enhanced_v2 | Full KCH-MedRank vs Pairwise graph LTR | MRR | 10 | 0.7866 | 0.7867 | +0.0001 | [-0.0046, +0.0048] | +0.01% | 0.9559 |
| bioasq_enhanced_v2 | Full KCH-MedRank vs Pairwise graph LTR | Recall | 10 | 0.5306 | 0.5329 | +0.0023 | [-0.0012, +0.0063] | +0.43% | 0.2034 |
| bioasq_enhanced_v2 | Full KCH-MedRank vs Pairwise graph LTR | Ndcg | 10 | 0.6409 | 0.6433 | +0.0024 | [-0.0001, +0.0050] | +0.38% | 0.0616 |
| bioasq_enhanced_v2 | Full KCH-MedRank vs Remove MeSH hierarchy | MRR | 10 | 0.7871 | 0.7867 | -0.0004 | [-0.0052, +0.0044] | -0.05% | 0.8683 |
| bioasq_enhanced_v2 | Full KCH-MedRank vs Remove MeSH hierarchy | Recall | 10 | 0.5332 | 0.5329 | -0.0003 | [-0.0029, +0.0022] | -0.05% | 0.8247 |
| bioasq_enhanced_v2 | Full KCH-MedRank vs Remove MeSH hierarchy | Ndcg | 10 | 0.6440 | 0.6433 | -0.0007 | [-0.0032, +0.0017] | -0.11% | 0.5731 |
| bioasq_enhanced_v2 | Full KCH-MedRank vs Remove hypergraph diffusion/centrality | MRR | 10 | 0.7880 | 0.7867 | -0.0012 | [-0.0064, +0.0039] | -0.16% | 0.6339 |
| bioasq_enhanced_v2 | Full KCH-MedRank vs Remove hypergraph diffusion/centrality | Recall | 10 | 0.5328 | 0.5329 | +0.0001 | [-0.0017, +0.0019] | +0.02% | 0.9603 |
| bioasq_enhanced_v2 | Full KCH-MedRank vs Remove hypergraph diffusion/centrality | Ndcg | 10 | 0.6453 | 0.6433 | -0.0020 | [-0.0045, +0.0004] | -0.31% | 0.1092 |

All tests use paired bootstrap over 943 held-out questions with 10,000 resamples.
