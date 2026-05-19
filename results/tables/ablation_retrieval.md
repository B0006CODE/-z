| method | split | recall@5 | recall@10 | recall@20 | mrr@10 | ndcg@10 | selected weights |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid RRF | test | 0.3863 | 0.4831 | 0.5562 | 0.7568 | 0.5959 | n/a |
| No-knowledge hypergraph | test | 0.3863 | 0.4831 | 0.5562 | 0.7568 | 0.5959 | hg=0.00, entity=0.00 |
| Pairwise graph | test | 0.3863 | 0.4833 | 0.5563 | 0.7569 | 0.5959 | hg=0.00, entity=0.02 |
| Knowledge hypergraph | test | 0.3867 | 0.4831 | 0.5563 | 0.7565 | 0.5958 | hg=0.02, entity=0.02 |
| Remove hypergraph diffusion | test | 0.3863 | 0.4833 | 0.5563 | 0.7569 | 0.5959 | hg=0.00, entity=0.02 |
| Remove biomedical entity overlap | test | 0.3863 | 0.4831 | 0.5562 | 0.7568 | 0.5959 | hg=0.02, entity=0.00 |

Interpretation: current dictionary-only medical knowledge features are weak. Validation tuning usually suppresses diffusion, and the strongest held-out result is effectively the Hybrid RRF baseline plus a very small entity-coverage term. The evidence supports improving knowledge mapping before making stronger claims about hypergraph reranking.
