| method | split | recall@5 | recall@10 | recall@20 | mrr@10 | ndcg@10 | selected weights |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid RRF | test | 0.3863 | 0.4831 | 0.5562 | 0.7568 | 0.5959 | n/a |
| MeSH overlap rerank | test | 0.3863 | 0.4831 | 0.5561 | 0.7568 | 0.5959 | mesh=0.02 |
| No-knowledge hypergraph | test | 0.3863 | 0.4831 | 0.5562 | 0.7568 | 0.5959 | hg=0.00, entity=0.00 |
| Pairwise graph | test | 0.3863 | 0.4833 | 0.5563 | 0.7569 | 0.5959 | hg=0.00, entity=0.02 |
| Knowledge hypergraph, no MeSH | test | 0.3867 | 0.4831 | 0.5563 | 0.7565 | 0.5958 | hg=0.02, entity=0.02, mesh=0.00 |
| Knowledge hypergraph + MeSH | test | 0.3866 | 0.4832 | 0.5564 | 0.7565 | 0.5957 | hg=0.02, entity=0.02, mesh=0.02 |
| Remove hypergraph diffusion | test | 0.3863 | 0.4833 | 0.5563 | 0.7569 | 0.5959 | hg=0.00, entity=0.02 |
| Remove biomedical entity overlap | test | 0.3863 | 0.4831 | 0.5562 | 0.7568 | 0.5959 | hg=0.02, entity=0.00 |

Interpretation: current dictionary-only entity features and lexical MeSH matching are weak. Adding MeSH as document-level hyperedges changes held-out metrics only marginally and does not beat Hybrid RRF on MRR@10 or nDCG@10. The evidence supports improving knowledge mapping or relation expansion before making stronger claims about hypergraph reranking.
