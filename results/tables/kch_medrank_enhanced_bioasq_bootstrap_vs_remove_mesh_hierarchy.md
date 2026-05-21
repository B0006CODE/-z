| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_enhanced | Remove MeSH hierarchy | Full KCH-MedRank | mrr | 10 | 0.7854 | 0.7882 | +0.0028 | [-0.0024, +0.0079] | +0.35% | 0.2938 |
| bioasq_enhanced | Remove MeSH hierarchy | Full KCH-MedRank | recall | 10 | 0.5355 | 0.5332 | -0.0023 | [-0.0075, +0.0026] | -0.43% | 0.3562 |
| bioasq_enhanced | Remove MeSH hierarchy | Full KCH-MedRank | ndcg | 10 | 0.6441 | 0.6446 | +0.0006 | [-0.0027, +0.0037] | +0.09% | 0.7407 |
| bioasq_enhanced | Remove MeSH hierarchy | Full KCH-MedRank | hit | 10 | 0.8823 | 0.8812 | -0.0011 | [-0.0064, +0.0042] | -0.12% | 0.8683 |

Paired bootstrap over 943 questions, 10000 resamples, seed=503.
