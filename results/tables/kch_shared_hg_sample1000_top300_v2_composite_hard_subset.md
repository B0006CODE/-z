| method | recall@5 | recall@10 | mrr@10 | ndcg@10 | evidence_coverage@10 | recall@100 | mrr@100 | ndcg@100 | delta_mrr@10 | delta_recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.0476 | 0.0635 | 0.0714 | 0.0636 | 0.0635 | 0.1459 | 0.0810 | 0.0976 |  |  |
| Dense | 0.1317 | 0.1317 | 0.3810 | 0.1661 | 0.1317 | 0.2902 | 0.3838 | 0.2369 |  |  |
| Hybrid RRF | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6139 | 0.0488 | 0.1993 |  |  |
| Biomedical semantic reranker only | 0.0000 | 0.0317 | 0.0179 | 0.0203 | 0.0317 | 0.6139 | 0.0451 | 0.1944 |  |  |
| Flat biomedical knowledge LambdaMART without graph structure | 0.1397 | 0.2189 | 0.4016 | 0.2193 | 0.2189 | 0.6918 | 0.4248 | 0.3789 |  |  |
| Retrieval + hypergraph structural LambdaMART | 0.0921 | 0.4284 | 0.3186 | 0.2495 | 0.4284 | 0.6998 | 0.3186 | 0.3556 |  |  |
| Pairwise graph LTR | 0.1317 | 0.2508 | 0.3036 | 0.2036 | 0.2508 | 0.7156 | 0.3248 | 0.3693 |  |  |
| Full KCH-MedRank | 0.1665 | 0.2777 | 0.4524 | 0.2637 | 0.2777 | 0.6918 | 0.4626 | 0.4017 |  |  |
| Remove hypergraph diffusion and centrality | 0.1397 | 0.3332 | 0.3968 | 0.2580 | 0.3332 | 0.6918 | 0.4063 | 0.3815 |  |  |
