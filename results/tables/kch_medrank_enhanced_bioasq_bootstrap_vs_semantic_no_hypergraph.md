| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_enhanced | LambdaMART + semantic, no hypergraph | Full KCH-MedRank | mrr | 10 | 0.7885 | 0.7882 | -0.0003 | [-0.0047, +0.0040] | -0.04% | 0.8947 |
| bioasq_enhanced | LambdaMART + semantic, no hypergraph | Full KCH-MedRank | recall | 10 | 0.5328 | 0.5332 | +0.0004 | [-0.0032, +0.0041] | +0.08% | 0.8253 |
| bioasq_enhanced | LambdaMART + semantic, no hypergraph | Full KCH-MedRank | ndcg | 10 | 0.6451 | 0.6446 | -0.0005 | [-0.0030, +0.0019] | -0.08% | 0.6895 |
| bioasq_enhanced | LambdaMART + semantic, no hypergraph | Full KCH-MedRank | hit | 10 | 0.8812 | 0.8812 | +0.0000 | [-0.0032, +0.0032] | +0.00% | 1.0000 |

Paired bootstrap over 943 questions, 10000 resamples, seed=501.
