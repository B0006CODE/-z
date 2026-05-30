| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_shared_hg_sample1000_top300_v2_composite | Pairwise_Graph_LTR | Full_KCH | mrr | 10 | 0.8659 | 0.8841 | +0.0182 | [+0.0037, +0.0350] | +2.11% | 0.0124 |
| bioasq_shared_hg_sample1000_top300_v2_composite | Pairwise_Graph_LTR | Full_KCH | recall | 10 | 0.5097 | 0.5066 | -0.0032 | [-0.0088, +0.0025] | -0.62% | 0.2807 |
| bioasq_shared_hg_sample1000_top300_v2_composite | Pairwise_Graph_LTR | Full_KCH | hit | 10 | 0.9800 | 0.9800 | +0.0000 | [-0.0150, +0.0150] | +0.00% | 1.0000 |
| bioasq_shared_hg_sample1000_top300_v2_composite | Pairwise_Graph_LTR | Full_KCH | ndcg | 10 | 0.6767 | 0.6762 | -0.0005 | [-0.0076, +0.0055] | -0.08% | 0.8746 |

Paired bootstrap over 200 questions, 5000 resamples, seed=42.
