| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_shared_hg_sample2000_top300_v2 | Pairwise graph LTR | NDCG-selected flat+strict KCH RRF | mrr | 10 | 0.8675 | 0.8726 | +0.0051 | [-0.0032, +0.0133] | +0.59% | 0.2304 |
| bioasq_shared_hg_sample2000_top300_v2 | Pairwise graph LTR | NDCG-selected flat+strict KCH RRF | recall | 10 | 0.5180 | 0.5201 | +0.0020 | [-0.0017, +0.0057] | +0.39% | 0.3103 |
| bioasq_shared_hg_sample2000_top300_v2 | Pairwise graph LTR | NDCG-selected flat+strict KCH RRF | ndcg | 10 | 0.6962 | 0.6972 | +0.0010 | [-0.0033, +0.0050] | +0.14% | 0.6295 |

Paired bootstrap over 400 questions, 5000 resamples, seed=46.
