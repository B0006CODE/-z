| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_enhanced | Pairwise graph LTR | Full KCH-MedRank | mrr | 10 | 0.7876 | 0.7882 | +0.0006 | [-0.0043, +0.0056] | +0.08% | 0.7991 |
| bioasq_enhanced | Pairwise graph LTR | Full KCH-MedRank | recall | 10 | 0.5292 | 0.5332 | +0.0040 | [+0.0010, +0.0079] | +0.76% | 0.0054 |
| bioasq_enhanced | Pairwise graph LTR | Full KCH-MedRank | ndcg | 10 | 0.6412 | 0.6446 | +0.0035 | [+0.0011, +0.0062] | +0.54% | 0.0048 |
| bioasq_enhanced | Pairwise graph LTR | Full KCH-MedRank | hit | 10 | 0.8791 | 0.8812 | +0.0021 | [+0.0000, +0.0053] | +0.24% | 0.2748 |

Paired bootstrap over 943 questions, 10000 resamples, seed=502.
