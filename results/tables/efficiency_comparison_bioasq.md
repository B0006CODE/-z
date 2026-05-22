| method | device | questions | candidates | rerank_seconds | ms_per_query | candidates_per_second | recall@10 | mrr@10 | ndcg@10 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MedCPT Cross-Encoder | cuda | 943 | 94300 | 699.46 | 741.74 | 134.82 | 0.5172 | 0.7775 | 0.6390 | tokenization + forward |
| KCH-MedRank | CPU | 943 | 94300 | 36.31 | 38.50 | 2597.07 | 0.5332 | 0.7882 | 0.6446 | features + LightGBM |
