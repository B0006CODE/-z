| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PubMedQA | Hybrid RRF | HGB all features | mrr | 10 | 0.9850 | 0.9861 | +0.0012 | [-0.0107, +0.0137] | +0.12% | 0.8783 |
| PubMedQA | Hybrid RRF | HGB all features | recall | 10 | 0.8200 | 0.8953 | +0.0754 | [+0.0509, +0.1012] | +9.19% | 0.0002 |

Paired bootstrap over 200 questions, 10000 resamples, seed=42.
