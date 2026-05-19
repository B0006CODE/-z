| method | split | recall@5 | recall@10 | recall@20 | mrr@10 | ndcg@10 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid RRF | held-out test | 0.3690 | 0.4636 | 0.5349 | 0.7550 | 0.5848 | Same test qids as supervised reranker |
| Logistic reranker | held-out test | 0.3708 | 0.4652 | 0.5377 | 0.7538 | 0.5866 | Validation-selected `C=0.1`, blend=0.35 |
| HistGradient reranker | held-out test | 0.3804 | 0.4730 | 0.5459 | 0.7696 | 0.6009 | Validation-selected `l2=0.01`, blend=0.10 |

Deterministic split: train qids = 2832, validation qids = 944, test qids = 943, using `question_id % 5` with validation remainder 3 and test remainder 4. The supervised reranker trains on top-100 Hybrid RRF candidates and uses hybrid, BM25, dense, entity, MeSH, PrimeKG, and local hypergraph features.
