| method | recall@5 | recall@10 | mrr@10 | ndcg@10 | evidence_coverage@10 | recall@100 | mrr@100 | ndcg@100 | delta_mrr@10 | delta_recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.0238 | 0.0460 | 0.0429 | 0.0388 | 0.0460 | 0.1872 | 0.0527 | 0.0950 |  |  |
| Dense | 0.0659 | 0.0659 | 0.1905 | 0.0831 | 0.0659 | 0.2308 | 0.1966 | 0.1430 |  |  |
| Hybrid RRF | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7427 | 0.0468 | 0.2071 |  |  |
| Biomedical semantic reranker only | 0.0143 | 0.0302 | 0.0446 | 0.0254 | 0.0302 | 0.7427 | 0.0656 | 0.1999 |  |  |
| Flat biomedical knowledge LambdaMART without graph structure | 0.2127 | 0.2683 | 0.2768 | 0.2021 | 0.2683 | 0.7888 | 0.3010 | 0.3558 |  |  |
| Retrieval + hypergraph structural LambdaMART | 0.2142 | 0.3396 | 0.2192 | 0.1996 | 0.3396 | 0.7982 | 0.2281 | 0.3348 |  |  |
| Pairwise graph LTR | 0.2365 | 0.2737 | 0.2821 | 0.2247 | 0.2737 | 0.7935 | 0.3010 | 0.3789 |  |  |
| Full KCH-MedRank | 0.2484 | 0.3095 | 0.2220 | 0.1954 | 0.3095 | 0.8007 | 0.2425 | 0.3425 |  |  |
| Remove hypergraph diffusion and centrality | 0.2008 | 0.2737 | 0.2312 | 0.1922 | 0.2737 | 0.7816 | 0.2488 | 0.3422 |  |  |
