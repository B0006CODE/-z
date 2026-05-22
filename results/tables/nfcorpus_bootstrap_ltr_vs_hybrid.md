| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BEIR NFCorpus | Hybrid RRF | Retrieval-feature LambdaMART | mrr | 10 | 0.5530 | 0.5725 | +0.0195 | [-0.0009, +0.0402] | +3.53% | 0.0592 |
| BEIR NFCorpus | Hybrid RRF | Retrieval-feature LambdaMART | recall | 10 | 0.1676 | 0.1770 | +0.0093 | [+0.0028, +0.0177] | +5.57% | 0.0034 |
| BEIR NFCorpus | Hybrid RRF | Retrieval-feature LambdaMART | ndcg | 10 | 0.3436 | 0.3627 | +0.0190 | [+0.0091, +0.0300] | +5.54% | 0.0002 |
| BEIR NFCorpus | Hybrid RRF | Retrieval-feature LambdaMART | hit | 10 | 0.7214 | 0.7461 | +0.0248 | [-0.0031, +0.0526] | +3.43% | 0.0832 |

Paired bootstrap over 323 questions, 10000 resamples, seed=42.
