| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_all_hg_sources_sample500_top300_sourcefeat | Flat knowledge LambdaMART | Hypergraph structural LambdaMART + source reasons | mrr | 10 | 0.8884 | 0.8992 | +0.0108 | [-0.0202, +0.0418] | +1.21% | 0.5039 |
| bioasq_all_hg_sources_sample500_top300_sourcefeat | Flat knowledge LambdaMART | Hypergraph structural LambdaMART + source reasons | mrr | 100 | 0.8893 | 0.8992 | +0.0099 | [-0.0217, +0.0404] | +1.11% | 0.5275 |
| bioasq_all_hg_sources_sample500_top300_sourcefeat | Flat knowledge LambdaMART | Hypergraph structural LambdaMART + source reasons | recall | 10 | 0.4790 | 0.4903 | +0.0113 | [-0.0032, +0.0271] | +2.37% | 0.1324 |
| bioasq_all_hg_sources_sample500_top300_sourcefeat | Flat knowledge LambdaMART | Hypergraph structural LambdaMART + source reasons | recall | 100 | 0.8145 | 0.8272 | +0.0127 | [+0.0016, +0.0247] | +1.56% | 0.0280 |
| bioasq_all_hg_sources_sample500_top300_sourcefeat | Flat knowledge LambdaMART | Hypergraph structural LambdaMART + source reasons | ndcg | 10 | 0.6437 | 0.6523 | +0.0086 | [-0.0050, +0.0235] | +1.33% | 0.2364 |
| bioasq_all_hg_sources_sample500_top300_sourcefeat | Flat knowledge LambdaMART | Hypergraph structural LambdaMART + source reasons | ndcg | 100 | 0.7240 | 0.7361 | +0.0121 | [+0.0021, +0.0238] | +1.67% | 0.0156 |

Paired bootstrap over 99 questions, 5000 resamples, seed=42.
