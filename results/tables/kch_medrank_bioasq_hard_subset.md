| method | recall@5 | recall@10 | mrr@10 | ndcg@10 | evidence_coverage@10 | recall@100 | mrr@100 | ndcg@100 | delta_mrr@10 | delta_recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.1149 | 0.1657 | 0.1679 | 0.1163 | 0.1657 | 0.5288 | 0.1971 | 0.2268 |  |  |
| Dense | 0.0266 | 0.0442 | 0.0712 | 0.0383 | 0.0442 | 0.3111 | 0.0914 | 0.1165 |  |  |
| Hybrid RRF | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5718 | 0.0492 | 0.1743 |  |  |
| Biomedical semantic reranker only | 0.0266 | 0.0442 | 0.0712 | 0.0383 | 0.0442 | 0.5718 | 0.0973 | 0.1708 |  |  |
| Retrieval-feature-only LambdaMART | 0.0547 | 0.1319 | 0.0539 | 0.0648 | 0.1319 | 0.5718 | 0.0841 | 0.1989 |  |  |
| LambdaMART + biomedical semantic without hypergraph | 0.0462 | 0.1447 | 0.0513 | 0.0662 | 0.1447 | 0.5718 | 0.0793 | 0.1969 |  |  |
| Pairwise graph LTR | 0.0462 | 0.1329 | 0.0374 | 0.0553 | 0.1329 | 0.5718 | 0.0712 | 0.1919 |  |  |
| Hypergraph LTR without medical knowledge | 0.0462 | 0.1276 | 0.0481 | 0.0589 | 0.1276 | 0.5718 | 0.0777 | 0.1948 |  |  |
| Full KCH-MedRank | 0.0462 | 0.1447 | 0.0506 | 0.0655 | 0.1447 | 0.5718 | 0.0773 | 0.1947 |  |  |
| Remove biomedical semantic reranker | 0.0462 | 0.0993 | 0.0399 | 0.0476 | 0.0993 | 0.5718 | 0.0741 | 0.1924 |  |  |
| Remove MeSH hierarchy features | 0.0462 | 0.1405 | 0.0487 | 0.0620 | 0.1405 | 0.5718 | 0.0758 | 0.1938 |  |  |
| Remove biomedical entity features | 0.0462 | 0.1447 | 0.0520 | 0.0661 | 0.1447 | 0.5718 | 0.0784 | 0.1954 |  |  |
| Remove hypergraph diffusion and centrality | 0.0462 | 0.1447 | 0.0531 | 0.0665 | 0.1447 | 0.5718 | 0.0818 | 0.1976 |  |  |
| Remove PrimeKG relation features | 0.0077 | 0.1405 | 0.0495 | 0.0616 | 0.1405 | 0.5718 | 0.0762 | 0.1931 |  |  |
