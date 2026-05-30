| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_shared_hg_sample2000_top300_v2 | Flat biomedical knowledge LTR | NDCG-selected flat+strict KCH RRF | mrr | 10 | 0.8719 | 0.8726 | +0.0008 | [-0.0006, +0.0022] | +0.09% | 0.3083 |
| bioasq_shared_hg_sample2000_top300_v2 | Flat biomedical knowledge LTR | NDCG-selected flat+strict KCH RRF | recall | 10 | 0.5173 | 0.5201 | +0.0028 | [-0.0003, +0.0065] | +0.53% | 0.0832 |
| bioasq_shared_hg_sample2000_top300_v2 | Flat biomedical knowledge LTR | NDCG-selected flat+strict KCH RRF | ndcg | 10 | 0.6958 | 0.6972 | +0.0013 | [-0.0007, +0.0033] | +0.19% | 0.1852 |

Paired bootstrap over 400 questions, 5000 resamples, seed=45.
