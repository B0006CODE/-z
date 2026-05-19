| Dataset | Comparison | Metric | k | Baseline | HGB | Delta | 95% CI | Relative | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BioASQ | HGB all features vs Hybrid RRF | MRR | 10 | 0.7550 | 0.7696 | +0.0146 | [+0.0066, +0.0228] | +1.94% | <0.001 |
| BioASQ | HGB all features vs Hybrid RRF | Recall | 10 | 0.4636 | 0.4730 | +0.0094 | [+0.0041, +0.0148] | +2.03% | <0.001 |
| PubMedQA | HGB all features vs Hybrid RRF | MRR | 10 | 0.9850 | 0.9861 | +0.0012 | [-0.0107, +0.0137] | +0.12% | 0.8783 |
| PubMedQA | HGB all features vs Hybrid RRF | Recall | 10 | 0.8200 | 0.8953 | +0.0754 | [+0.0509, +0.1012] | +9.19% | <0.001 |

All tests use paired bootstrap over question-level metric values with 10,000 resamples.
