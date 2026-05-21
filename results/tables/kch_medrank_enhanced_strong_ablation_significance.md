| Dataset | Comparison | Metric | k | Baseline | Candidate | Delta | 95% CI | Relative | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_enhanced | Full KCH-MedRank vs LambdaMART + semantic, no hypergraph | MRR | 10 | 0.7885 | 0.7882 | -0.0003 | [-0.0047, +0.0040] | -0.04% | 0.8947 |
| bioasq_enhanced | Full KCH-MedRank vs LambdaMART + semantic, no hypergraph | Recall | 10 | 0.5328 | 0.5332 | +0.0004 | [-0.0032, +0.0041] | +0.08% | 0.8253 |
| bioasq_enhanced | Full KCH-MedRank vs LambdaMART + semantic, no hypergraph | Ndcg | 10 | 0.6451 | 0.6446 | -0.0005 | [-0.0030, +0.0019] | -0.08% | 0.6895 |
| bioasq_enhanced | Full KCH-MedRank vs LambdaMART + semantic, no hypergraph | Hit | 10 | 0.8812 | 0.8812 | +0.0000 | [-0.0032, +0.0032] | +0.00% | 1.0000 |
| bioasq_enhanced | Full KCH-MedRank vs Pairwise graph LTR | MRR | 10 | 0.7876 | 0.7882 | +0.0006 | [-0.0043, +0.0056] | +0.08% | 0.7991 |
| bioasq_enhanced | Full KCH-MedRank vs Pairwise graph LTR | Recall | 10 | 0.5292 | 0.5332 | +0.0040 | [+0.0010, +0.0079] | +0.76% | 0.0054 |
| bioasq_enhanced | Full KCH-MedRank vs Pairwise graph LTR | Ndcg | 10 | 0.6412 | 0.6446 | +0.0035 | [+0.0011, +0.0062] | +0.54% | 0.0048 |
| bioasq_enhanced | Full KCH-MedRank vs Pairwise graph LTR | Hit | 10 | 0.8791 | 0.8812 | +0.0021 | [+0.0000, +0.0053] | +0.24% | 0.2748 |
| bioasq_enhanced | Full KCH-MedRank vs Remove MeSH hierarchy | MRR | 10 | 0.7854 | 0.7882 | +0.0028 | [-0.0024, +0.0079] | +0.35% | 0.2938 |
| bioasq_enhanced | Full KCH-MedRank vs Remove MeSH hierarchy | Recall | 10 | 0.5355 | 0.5332 | -0.0023 | [-0.0075, +0.0026] | -0.43% | 0.3562 |
| bioasq_enhanced | Full KCH-MedRank vs Remove MeSH hierarchy | Ndcg | 10 | 0.6441 | 0.6446 | +0.0006 | [-0.0027, +0.0037] | +0.09% | 0.7407 |
| bioasq_enhanced | Full KCH-MedRank vs Remove MeSH hierarchy | Hit | 10 | 0.8823 | 0.8812 | -0.0011 | [-0.0064, +0.0042] | -0.12% | 0.8683 |
| bioasq_enhanced | Full KCH-MedRank vs Remove hypergraph diffusion/centrality | MRR | 10 | 0.7872 | 0.7882 | +0.0010 | [-0.0040, +0.0059] | +0.13% | 0.6855 |
| bioasq_enhanced | Full KCH-MedRank vs Remove hypergraph diffusion/centrality | Recall | 10 | 0.5341 | 0.5332 | -0.0009 | [-0.0049, +0.0026] | -0.16% | 0.6777 |
| bioasq_enhanced | Full KCH-MedRank vs Remove hypergraph diffusion/centrality | Ndcg | 10 | 0.6452 | 0.6446 | -0.0006 | [-0.0030, +0.0018] | -0.09% | 0.6421 |
| bioasq_enhanced | Full KCH-MedRank vs Remove hypergraph diffusion/centrality | Hit | 10 | 0.8834 | 0.8812 | -0.0021 | [-0.0053, +0.0000] | -0.24% | 0.2782 |

All tests use paired bootstrap over question-level metric values with 10,000 resamples.
