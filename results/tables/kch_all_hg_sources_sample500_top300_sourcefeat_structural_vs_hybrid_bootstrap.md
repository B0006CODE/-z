| Dataset | Baseline | Candidate | Metric | k | Baseline Score | Candidate Score | Delta | 95% CI | Rel. Delta | p-value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq_all_hg_sources_sample500_top300_sourcefeat | Hybrid/expanded candidate order | Hypergraph structural LambdaMART + source reasons | mrr | 10 | 0.8966 | 0.8992 | +0.0026 | [-0.0230, +0.0278] | +0.29% | 0.8406 |
| bioasq_all_hg_sources_sample500_top300_sourcefeat | Hybrid/expanded candidate order | Hypergraph structural LambdaMART + source reasons | mrr | 100 | 0.8966 | 0.8992 | +0.0026 | [-0.0231, +0.0286] | +0.29% | 0.8718 |
| bioasq_all_hg_sources_sample500_top300_sourcefeat | Hybrid/expanded candidate order | Hypergraph structural LambdaMART + source reasons | recall | 10 | 0.4725 | 0.4903 | +0.0179 | [+0.0006, +0.0368] | +3.78% | 0.0420 |
| bioasq_all_hg_sources_sample500_top300_sourcefeat | Hybrid/expanded candidate order | Hypergraph structural LambdaMART + source reasons | recall | 100 | 0.7590 | 0.8272 | +0.0682 | [+0.0402, +0.0962] | +8.99% | 0.0004 |
| bioasq_all_hg_sources_sample500_top300_sourcefeat | Hybrid/expanded candidate order | Hypergraph structural LambdaMART + source reasons | ndcg | 10 | 0.6402 | 0.6523 | +0.0121 | [-0.0013, +0.0270] | +1.90% | 0.0752 |
| bioasq_all_hg_sources_sample500_top300_sourcefeat | Hybrid/expanded candidate order | Hypergraph structural LambdaMART + source reasons | ndcg | 100 | 0.6943 | 0.7361 | +0.0418 | [+0.0272, +0.0570] | +6.02% | 0.0004 |

Paired bootstrap over 99 questions, 5000 resamples, seed=42.
