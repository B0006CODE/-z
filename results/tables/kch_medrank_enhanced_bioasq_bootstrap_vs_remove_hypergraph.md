| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_enhanced | Remove hypergraph diffusion/centrality | Full KCH-MedRank | mrr | 10 | 0.7872 | 0.7882 | +0.0010 | [-0.0040, +0.0059] | +0.13% | 0.6855 |
| bioasq_enhanced | Remove hypergraph diffusion/centrality | Full KCH-MedRank | recall | 10 | 0.5341 | 0.5332 | -0.0009 | [-0.0049, +0.0026] | -0.16% | 0.6777 |
| bioasq_enhanced | Remove hypergraph diffusion/centrality | Full KCH-MedRank | ndcg | 10 | 0.6452 | 0.6446 | -0.0006 | [-0.0030, +0.0018] | -0.09% | 0.6421 |
| bioasq_enhanced | Remove hypergraph diffusion/centrality | Full KCH-MedRank | hit | 10 | 0.8834 | 0.8812 | -0.0021 | [-0.0053, +0.0000] | -0.24% | 0.2782 |

Paired bootstrap over 943 questions, 10000 resamples, seed=504.
