# Enhanced Fusion Validation Selection

Fusion weights were selected on the deterministic validation split only (`qid % 5 == 3`). The held-out test split (`qid % 5 == 4`) is reported after selection.

| fusion run | BM25 | dense | fielded BM25 | MedCPT dual encoder | validation recall@100 | validation mrr@10 | test recall@100 | test mrr@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original Hybrid | 1.0 | 1.0 | 0.0 | 0.0 | 0.6325 | 0.7564 | 0.6077 | 0.7550 |
| w052 | 1.0 | 1.0 | 0.5 | 0.2 | 0.6391 | 0.7587 | 0.6163 | 0.7593 |
| w084 | 1.0 | 1.0 | 0.8 | 0.4 | 0.7329 | 0.7577 | 0.7084 | 0.7602 |
| w102 | 1.0 | 1.0 | 1.0 | 0.2 | 0.7529 | 0.7516 | 0.7285 | 0.7542 |
| w122, selected | 1.0 | 1.0 | 1.2 | 0.2 | 0.7568 | 0.7533 | 0.7388 | 0.7530 |

The selected run maximizes validation Recall@100, matching the goal of improving the reranker candidate ceiling rather than directly optimizing first-stage MRR.
