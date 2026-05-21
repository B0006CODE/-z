# Efficiency and Computational Profile

The available logs record dataset scale, candidate-pool size, feature dimensionality, knowledge-source coverage, and hardware notes. They do not provide reliable end-to-end wall-clock time for every component, so this table reports a computational profile rather than fabricated runtime numbers.

| Component | Recorded scale or setting | Computational implication |
| --- | --- | --- |
| BioASQ processed data | 4,719 questions; 40,221 passages; 42,608 qrels | Small enough for reproducible offline reranking experiments. |
| Candidate pool | top-100 passages per question; 471,900 BioASQ prediction rows | Reranking is bounded by local candidate pools, not by a global graph over the full corpus. |
| Split used for enhanced KCH-MedRank | 2,832 train questions; 944 validation questions; 943 held-out test questions | Hyperparameters are selected before the held-out test evaluation. |
| Full KCH-MedRank feature matrix | 45 features; 283,200 training rows; 377,600 final train+validation rows | Learning-to-rank remains a tabular reranking problem over query groups. |
| Local hypergraph scope | per-question graph over question, top-100 candidates, document, entity, MeSH, hierarchy, and optional relation nodes | Memory and propagation cost scale with top-M and local feature coverage rather than corpus size. |
| MeSH coverage | 31,110 descriptors; 31,108 descriptors with tree numbers; 37,356 / 40,221 passages with MeSH; 4,264 / 4,719 questions with synonym-aware MeSH matches | Hierarchy-aware features are broadly available and suitable as the main controlled-vocabulary constraint. |
| PrimeKG filtering | 8,100,498 rows scanned; 5,766 project-local relations retained; 1,385 matched local entities | Relation features are sparse and should remain auxiliary. |
| Dense biomedical retrieval | MedCPT dual encoder run with CUDA on NVIDIA GeForce RTX 4060 Laptop GPU; top-100 output for 4,719 questions | GPU acceleration is practical for dense biomedical retrieval; CPU-only reproduction may be slower. |
| Cross-encoder scoring | BioASQ CPU full run timed out after 2,400 seconds; PubMedQA MedCPT CPU smoke test timed out after 300 seconds | Biomedical cross-encoder baselines should use GPU or be restricted to controlled timing subsets. |
| Proposed reranking profile | Uses stored retrieval and semantic scores plus local hypergraph/LambdaMART features | The reranker is lightweight relative to full cross-encoder rescoring of every question-candidate pair. |
