| method | recall@5 | recall@10 | mrr@10 | ndcg@10 | evidence_coverage@10 | recall@100 | mrr@100 | ndcg@100 | delta_mrr@10 | delta_recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.0238 | 0.0460 | 0.0429 | 0.0388 | 0.0460 | 0.1872 | 0.0527 | 0.0950 |  |  |
| Dense | 0.0484 | 0.0627 | 0.1875 | 0.0727 | 0.0627 | 0.1745 | 0.1971 | 0.1120 |  |  |
| Hybrid RRF | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7427 | 0.0468 | 0.2071 |  |  |
| Biomedical semantic reranker only | 0.0143 | 0.0302 | 0.0446 | 0.0254 | 0.0302 | 0.7427 | 0.0656 | 0.1999 |  |  |
| Flat biomedical knowledge LambdaMART without graph structure | 0.2127 | 0.2683 | 0.2768 | 0.2021 | 0.2683 | 0.7888 | 0.3010 | 0.3558 |  |  |
| Strict-specificity KCH-MedRank without global structural counts | 0.2063 | 0.3174 | 0.2745 | 0.2120 | 0.3174 | 0.7102 | 0.2945 | 0.3365 |  |  |
