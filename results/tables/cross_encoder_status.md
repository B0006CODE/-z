| dataset | run | status | key result |
| --- | --- | --- | --- |
| PubMedQA pqa_labeled | full Hybrid top100 -> cross-encoder | completed | MRR@10 0.9783 -> 0.9806; Recall@10 0.8297 -> 0.8268 |
| PubMedQA pqa_labeled | Hybrid top100 -> HGB held-out test | completed | MRR@10 0.9850 -> 0.9861; Recall@10 0.8200 -> 0.8953 |
| BioASQ | Hybrid top100 -> HGB held-out test | completed earlier | MRR@10 0.7550 -> 0.7696 |
| BioASQ | Hybrid sample10 -> cross-encoder | completed | MRR@10 0.7833 -> 0.6250 |
| BioASQ | Hybrid held-out top100 -> cross-encoder | timed out on CPU | 2400s timeout; use GPU or biomedical cross-encoder for full run |
