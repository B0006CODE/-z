| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_shared_hg_sample2000_top300_v2 | Pairwise graph LTR | Strict KCH no global counts | mrr | 10 | 0.8675 | 0.8693 | +0.0018 | [-0.0068, +0.0103] | +0.21% | 0.6803 |
| bioasq_shared_hg_sample2000_top300_v2 | Pairwise graph LTR | Strict KCH no global counts | recall | 10 | 0.5180 | 0.5171 | -0.0009 | [-0.0059, +0.0040] | -0.18% | 0.6947 |
| bioasq_shared_hg_sample2000_top300_v2 | Pairwise graph LTR | Strict KCH no global counts | ndcg | 10 | 0.6962 | 0.6942 | -0.0020 | [-0.0067, +0.0024] | -0.29% | 0.4063 |

Paired bootstrap over 400 questions, 5000 resamples, seed=42.
