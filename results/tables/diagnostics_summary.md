| diagnostic | value |
| --- | --- |
| Hybrid top100 question hit rate | 0.9010 |
| Hybrid MRR@10 | 0.7686 |
| Oracle-within-top100 MRR@10 | 0.9010 |
| Hybrid Recall@10 | 0.4885 |
| Oracle-within-top100 Recall@10 | 0.5931 |
| Entity overlap AUC | 0.5756 |
| Question entity coverage AUC | 0.5762 |
| MeSH overlap AUC | 0.6209 |
| Question MeSH coverage AUC | 0.6210 |
| Questions with lexical MeSH matches | 3375 / 4719 |
| Passages with PubMed MeSH | 37356 / 40221 |
| Filtered PrimeKG relation rows | 5766 |
| Questions with PrimeKG relation in top100 | 330 / 4719 |
| PrimeKG relation-count AUC | 0.5094 |

Interpretation: the retrieval candidate set has enough headroom for reranking, but the original dictionary entity features are weak. PubMed MeSH metadata is substantially better than dictionary entities, though a simple overlap-only reranker still gives only tiny held-out changes. Exact-name PrimeKG matching is too sparse to help current reranking.
