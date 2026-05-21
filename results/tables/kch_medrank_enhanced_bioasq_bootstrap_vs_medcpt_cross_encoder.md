| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_enhanced | MedCPT-Cross-Encoder | Full-KCH-MedRank | mrr | 10 | 0.7775 | 0.7882 | +0.0107 | [-0.0036, +0.0254] | +1.38% | 0.1384 |
| bioasq_enhanced | MedCPT-Cross-Encoder | Full-KCH-MedRank | recall | 10 | 0.5172 | 0.5332 | +0.0160 | [+0.0044, +0.0283] | +3.09% | 0.0074 |
| bioasq_enhanced | MedCPT-Cross-Encoder | Full-KCH-MedRank | ndcg | 10 | 0.6390 | 0.6446 | +0.0056 | [-0.0042, +0.0154] | +0.88% | 0.2668 |

Paired bootstrap over 943 questions, 10000 resamples, seed=42.
