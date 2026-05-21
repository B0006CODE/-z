# Enhanced First-Stage Retrieval

BioASQ full set, 4,719 questions. The enhanced runs are candidate-generation variants used to raise the top-100 ceiling before KCH-MedRank reranking.

| method | recall@10 | mrr@10 | ndcg@10 | recall@100 | mrr@100 | ndcg@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid RRF, original BM25 + dense | 0.4885 | 0.7686 | 0.6031 | 0.6336 | 0.7701 | 0.6049 |
| Fielded BM25 + MeSH expansion | 0.4373 | 0.6530 | 0.5129 | 0.6966 | 0.6575 | 0.5710 |
| MedCPT dual-encoder dense | 0.4313 | 0.6797 | 0.5187 | 0.6158 | 0.6827 | 0.5425 |
| Enhanced Hybrid, balanced weights | 0.5020 | 0.7722 | 0.6146 | 0.7292 | 0.7740 | 0.6415 |
| Enhanced Hybrid, recall-optimized weights | 0.4915 | 0.7632 | 0.6019 | 0.7534 | 0.7656 | 0.6445 |

Balanced weights use BM25=1.0, dense=1.0, fielded BM25=0.8, MedCPT dual encoder=0.4. Recall-optimized weights use BM25=1.0, dense=1.0, fielded BM25=1.2, MedCPT dual encoder=0.2. The recall-optimized run is used as the candidate pool for the enhanced KCH-MedRank experiment.
