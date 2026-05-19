| method | features | recall@5 | recall@10 | recall@20 | mrr@10 | ndcg@10 | Δmrr@10 vs Hybrid | Δrecall@10 vs Hybrid |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| Hybrid RRF | n/a | 0.3690 | 0.4636 | 0.5349 | 0.7550 | 0.5848 | 0.0000 | 0.0000 |
| HGB reranker, all features | 23 | 0.3804 | 0.4730 | 0.5459 | 0.7696 | 0.6009 | +0.0146 | +0.0094 |
| HGB reranker, retrieval only | 7 | 0.3792 | 0.4723 | 0.5436 | 0.7633 | 0.5971 | +0.0083 | +0.0088 |
| HGB reranker, remove entity features | 19 | 0.3811 | 0.4718 | 0.5456 | 0.7665 | 0.5986 | +0.0114 | +0.0082 |
| HGB reranker, remove MeSH features | 18 | 0.3814 | 0.4728 | 0.5451 | 0.7672 | 0.5994 | +0.0122 | +0.0093 |
| HGB reranker, remove PrimeKG features | 20 | 0.3812 | 0.4740 | 0.5455 | 0.7679 | 0.6005 | +0.0129 | +0.0104 |
| HGB reranker, remove hypergraph features | 17 | 0.3813 | 0.4730 | 0.5436 | 0.7624 | 0.5987 | +0.0074 | +0.0094 |

All supervised runs use the same deterministic split: train qids = 2832, validation qids = 944, test qids = 943. Hyperparameters and blend weights are selected on validation only. The all-feature model gives the best held-out MRR@10 and nDCG@10. Removing hypergraph-derived features causes the largest MRR@10 drop among the leave-one-out settings, while removing PrimeKG slightly improves Recall@10 but lowers MRR@10, consistent with the earlier sparsity diagnostics.
