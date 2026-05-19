| top-M | method | recall@5 | recall@10 | recall@20 | recall@50 | mrr@10 | ndcg@10 | Δmrr@10 vs Hybrid |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | Hybrid RRF | 0.3690 | 0.4636 | 0.5349 | n/a | 0.7550 | 0.5848 | 0.0000 |
| 20 | HGB reranker, all features | 0.3797 | 0.4707 | 0.5349 | n/a | 0.7637 | 0.5971 | +0.0087 |
| 50 | Hybrid RRF | 0.3690 | 0.4636 | 0.5349 | 0.5887 | 0.7550 | 0.5848 | 0.0000 |
| 50 | HGB reranker, all features | 0.3798 | 0.4748 | 0.5468 | 0.5887 | 0.7662 | 0.5992 | +0.0112 |
| 100 | Hybrid RRF | 0.3690 | 0.4636 | 0.5349 | 0.5887 | 0.7550 | 0.5848 | 0.0000 |
| 100 | HGB reranker, all features | 0.3804 | 0.4730 | 0.5459 | 0.5910 | 0.7696 | 0.6009 | +0.0146 |

All rows use the same held-out test split with 943 questions. The learning reranker is retrained and validation-tuned separately for each candidate depth. Improvements over Hybrid RRF are stable for top-M 20, 50, and 100, with the largest MRR@10 gain at top-M 100.
