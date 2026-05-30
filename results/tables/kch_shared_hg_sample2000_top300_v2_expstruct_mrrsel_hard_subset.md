| method | recall@5 | recall@10 | mrr@10 | ndcg@10 | evidence_coverage@10 | recall@100 | mrr@100 | ndcg@100 | delta_mrr@10 | delta_recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.0238 | 0.0460 | 0.0429 | 0.0388 | 0.0460 | 0.1872 | 0.0527 | 0.0950 |  |  |
| Dense | 0.0484 | 0.0627 | 0.1875 | 0.0727 | 0.0627 | 0.1745 | 0.1971 | 0.1120 |  |  |
| Hybrid RRF | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7427 | 0.0468 | 0.2071 |  |  |
| Biomedical semantic reranker only | 0.0143 | 0.0302 | 0.0446 | 0.0254 | 0.0302 | 0.7427 | 0.0656 | 0.1999 |  |  |
| Full KCH-MedRank without global structural counts | 0.1770 | 0.3730 | 0.2097 | 0.2060 | 0.3730 | 0.7888 | 0.2250 | 0.3310 |  |  |
| Full KCH-MedRank rank-gated structural features | 0.1770 | 0.3174 | 0.1780 | 0.1886 | 0.3174 | 0.7935 | 0.1979 | 0.3275 |  |  |
