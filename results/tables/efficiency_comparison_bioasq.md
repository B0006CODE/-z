| method | device | uses_ce_score | online_ce | questions | candidates | rerank_seconds | ms_per_query | candidates_per_second | recall@10 | mrr@10 | ndcg@10 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MedCPT Cross-Encoder | cuda | yes | yes | 943 | 94300 | 699.46 | 741.74 | 134.82 | 0.5172 | 0.7775 | 0.6390 | tokenization + forward |
| KCH-MedRank | CPU | no | no | 943 | 94300 | 37.59 | 39.86 | 2508.70 | 0.5332 | 0.7882 | 0.6446 | features + LightGBM; semantic=dense_or_dual_encoder_predictions |
| KCH-MedRank without semantic feature | CPU | no | no | 943 | 94300 | 37.50 | 39.77 | 2514.47 | 0.5244 | 0.7790 | 0.6330 | knowledge/hypergraph features + LightGBM; semantic columns removed |
| Retrieval-feature-only LambdaMART | CPU | no | no | 943 | 94300 | 0.84 | 0.89 | 112905.71 | 0.5208 | 0.7791 | 0.6282 | retrieval metadata features + LightGBM |
