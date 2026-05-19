| method | split | recall@5 | recall@10 | recall@20 | recall@50 | recall@100 | mrr@10 | ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid RRF | test | 0.3863 | 0.4831 | 0.5562 | 0.6110 | 0.6290 | 0.7568 | 0.5959 |
| Local Hypergraph Rerank | test | 0.3867 | 0.4831 | 0.5563 | 0.6107 | 0.6290 | 0.7565 | 0.5958 |

Validation tuning selected `base_weight=1.0`, `hypergraph_weight=0.02`, and `entity_weight=0.02`.

Interpretation: the first local hypergraph reranker does not improve the strong Hybrid RRF baseline on the held-out test split. It gives tiny recall changes but slightly lowers MRR@10 and nDCG@10, so it should be treated as a diagnostic baseline before adding stronger medical knowledge constraints.
