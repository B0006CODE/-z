| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_shared_hg_sample1000_top300_v2_composite | Remove_Diffusion_Centrality | Full_KCH | mrr | 10 | 0.8670 | 0.8841 | +0.0171 | [+0.0026, +0.0336] | +1.97% | 0.0200 |
| bioasq_shared_hg_sample1000_top300_v2_composite | Remove_Diffusion_Centrality | Full_KCH | recall | 10 | 0.5038 | 0.5066 | +0.0027 | [-0.0068, +0.0120] | +0.55% | 0.5627 |
| bioasq_shared_hg_sample1000_top300_v2_composite | Remove_Diffusion_Centrality | Full_KCH | hit | 10 | 0.9850 | 0.9800 | -0.0050 | [-0.0150, +0.0000] | -0.51% | 0.7235 |
| bioasq_shared_hg_sample1000_top300_v2_composite | Remove_Diffusion_Centrality | Full_KCH | ndcg | 10 | 0.6735 | 0.6762 | +0.0027 | [-0.0050, +0.0097] | +0.40% | 0.4751 |

Paired bootstrap over 200 questions, 5000 resamples, seed=42.
