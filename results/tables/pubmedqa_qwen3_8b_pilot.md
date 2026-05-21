| method | num_eval | accuracy | macro_f1 | citation_support | unsupported_claim | entity_evaluable | entity_consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid RRF + Qwen3-8B | 100 | 0.3400 | 0.3506 | 0.9900 | 0.0100 | 100 | 0.5594 |
| Semantic cross-encoder + Qwen3-8B | 100 | 0.3400 | 0.3504 | 0.9900 | 0.0100 | 100 | 0.5782 |
| KCH-style hypergraph reranker + Qwen3-8B | 100 | 0.3700 | 0.3760 | 0.9900 | 0.0100 | 100 | 0.5751 |

Pilot diagnostic: fixed prompt, top-k evidence, deterministic decoding, and no model fine-tuning. Citation support requires at least one model-cited passage to overlap a PubMedQA gold evidence passage; unsupported claim rate is the complement of this citation-support test.
