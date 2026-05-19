| dataset | method | mrr@10 | recall@10 | ndcg@10 | delta_mrr@10 | delta_recall@10 | delta_ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BioASQ validation | Hybrid source | 0.7564 | 0.4907 | 0.6001 |  |  |  |
| BioASQ validation | Base HGB features | 0.7757 | 0.5005 | 0.6166 | +0.0193 | +0.0098 | +0.0166 |
| BioASQ validation | Enhanced norm+MeSH features | 0.7788 | 0.5061 | 0.6257 | +0.0224 | +0.0154 | +0.0256 |
| PubMedQA validation | Hybrid source | 0.9819 | 0.8413 | 0.8471 |  |  |  |
| PubMedQA validation | Base HGB features | 0.9918 | 0.9220 | 0.9178 | +0.0099 | +0.0808 | +0.0707 |
| PubMedQA validation | Enhanced norm+MeSH features | 0.9918 | 0.9279 | 0.9229 | +0.0099 | +0.0867 | +0.0759 |

Validation-only feasibility result. These values are not a replacement for the held-out test results; they indicate that lightweight entity normalization and MeSH alias/major-topic features are worth a controlled test-set run after freezing this feature set.
