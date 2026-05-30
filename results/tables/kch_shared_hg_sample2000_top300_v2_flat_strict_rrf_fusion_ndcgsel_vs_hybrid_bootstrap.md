| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_shared_hg_sample2000_top300_v2 | Hybrid RRF | NDCG-selected flat+strict KCH RRF | mrr | 10 | 0.8509 | 0.8726 | +0.0217 | [+0.0024, +0.0411] | +2.55% | 0.0276 |
| bioasq_shared_hg_sample2000_top300_v2 | Hybrid RRF | NDCG-selected flat+strict KCH RRF | recall | 10 | 0.4485 | 0.5201 | +0.0715 | [+0.0578, +0.0860] | +15.95% | 0.0004 |
| bioasq_shared_hg_sample2000_top300_v2 | Hybrid RRF | NDCG-selected flat+strict KCH RRF | ndcg | 10 | 0.6337 | 0.6972 | +0.0635 | [+0.0511, +0.0757] | +10.02% | 0.0004 |

Paired bootstrap over 400 questions, 5000 resamples, seed=47.
