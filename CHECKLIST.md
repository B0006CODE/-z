# Project Checklist

Project: Knowledge-Constrained Hypergraph Retrieval-Augmented Generation for Evidence-Grounded Medical Question Answering

Status legend:

- [ ] Pending
- [x] Done

## 0. Project Grounding

- [x] Inspect the project directory before editing.
- [x] Read `AGENTS.md` and confirm this is a new independent project.
- [x] Confirm initial scope: build the pipeline from scratch and start with first-stage retrieval.
- [x] Confirm primary dataset: `rag-datasets/rag-mini-bioasq`.
- [x] Confirm first implementation target: data preparation -> BM25 retrieval -> sample retrieval evaluation.
- [x] Keep this checklist updated after each completed task.

## 1. Repository Skeleton

- [x] Create `README.md` with project goal, scope, pipeline, and first runnable commands.
- [x] Create `requirements.txt` with minimal reproducible dependencies.
- [x] Create `configs/default.yaml` for paths, dataset name, retrieval settings, and random seed.
- [x] Create directory structure:
  - [x] `configs/`
  - [x] `data/raw/`
  - [x] `data/processed/`
  - [x] `data/external_knowledge/`
  - [x] `indexes/bm25/`
  - [x] `indexes/dense/`
  - [x] `outputs/retrieval/`
  - [x] `outputs/rerank/`
  - [x] `outputs/generation/`
  - [x] `results/tables/`
  - [x] `results/figures/`
  - [x] `results/metrics/`
  - [x] `scripts/`
  - [x] `src/data/`
  - [x] `src/retrieval/`
  - [x] `src/hypergraph/`
  - [x] `src/rerank/`
  - [x] `src/generation/`
  - [x] `src/evaluation/`
  - [x] `logs/`
- [x] Add package marker files where needed.

## 2. Data Loading: `rag-mini-bioasq`

- [x] Implement `scripts/prepare_bioasq.py`.
- [x] Load the Hugging Face dataset without assuming local cache.
- [x] Inspect available splits, columns, corpus format, question format, answers, and evidence labels.
- [x] Normalize question records.
- [x] Normalize corpus passage records.
- [x] Normalize gold evidence mappings.
- [x] Save processed questions to `data/processed/bioasq_questions.jsonl`.
- [x] Save processed corpus to `data/processed/bioasq_corpus.jsonl`.
- [x] Save processed qrels or evidence labels to `data/processed/bioasq_qrels.jsonl`.
- [x] Log dataset size, number of questions, number of passages, and evidence-label coverage.
- [x] Add a `--sample-size` option for a 10-question sanity run.
- [x] Verify the 10-question processed sample can be loaded back successfully.

## 3. First-Stage Retrieval: BM25 Baseline

- [x] Implement BM25 retriever module under `src/retrieval/`.
- [x] Implement `scripts/run_bm25.py`.
- [x] Support CLI arguments for config path, corpus path, question path, output path, top-k, and sample limit.
- [x] Build or load BM25 index under `indexes/bm25/`.
- [x] Save retrieval predictions as JSONL or CSV under `outputs/retrieval/`.
- [x] Include query id, passage id, rank, score, and retriever metadata in predictions.
- [x] Run BM25 on 10 questions with top-M values sufficient for initial evaluation.

## 4. Retrieval Evaluation

- [x] Implement retrieval metric utilities under `src/evaluation/`.
- [x] Implement `scripts/evaluate_retrieval.py`.
- [x] Compute Recall@k.
- [x] Compute Hit@k.
- [x] Compute MRR.
- [x] Compute nDCG@k if graded relevance is available.
- [x] Save metrics as JSON under `results/metrics/`.
- [x] Run evaluation for the 10-question BM25 sample.
- [x] Inspect sample predictions manually for obvious ID mismatch or data leakage.
- [x] Only after the sample looks reasonable, run the full BM25 baseline.

## 5. Dense And Hybrid Retrieval

- [x] Choose a default embedding model, preferably biomedical if practical.
- [x] Implement dense indexing and retrieval under `src/retrieval/`.
- [x] Save dense indexes under `indexes/dense/`.
- [x] Implement dense retrieval script or mode.
- [x] Evaluate dense retrieval with Recall@k, Hit@k, and MRR.
- [x] Implement hybrid BM25 + dense retrieval using Reciprocal Rank Fusion.
- [x] Evaluate hybrid retrieval for top-M values: 20, 50, 100.
- [x] Confirm first-stage recall is high enough before reranking.

## 6. Entity And Knowledge Processing

- [x] Implement biomedical entity extraction for questions and passages.
- [x] Implement simple normalization for entity strings.
- [x] Add MeSH concept or dictionary-based mapping if available.
- [x] Fetch PubMed MeSH metadata for corpus passages using passage ids as PMIDs.
- [x] Build question and passage MeSH feature files.
- [x] Load PrimeKG or a small relation table only when license and source are clear.
- [x] Save extracted entity features under `data/processed/`.
- [x] Save MeSH features under `data/processed/`.
- [x] Save relation features under `data/external_knowledge/` or `data/processed/`.
- [x] Measure feature sparsity for question entities, passage entities, MeSH overlap, and PrimeKG expansion.

## 7. Local Hypergraph Reranking

- [x] Implement local hypergraph construction under `src/hypergraph/`.
- [x] Include question nodes, passage nodes, entity nodes, concept nodes, and document nodes where available.
- [x] Implement question-entity hyperedges.
- [x] Implement passage-entity hyperedges.
- [x] Implement document-MeSH hyperedges when MeSH labels are available.
- [x] Implement PrimeKG relation hyperedges when relation data is available.
- [x] Implement question-passage candidate hyperedges.
- [x] Implement shared-entity hyperedges among candidate passages.
- [x] Implement hypergraph proximity or diffusion feature.
- [x] Implement reranking module under `src/rerank/`.
- [x] Tune coefficients or lightweight model on validation data, not on test results.
- [x] Save reranked predictions under `outputs/rerank/`.

## 8. Baselines And Ablations

- [x] BM25-only baseline.
- [x] Dense-only baseline.
- [x] Hybrid BM25 + dense baseline.
- [x] Medical entity overlap reranker baseline.
- [x] MeSH overlap reranker baseline.
- [x] Pairwise graph reranker without hyperedges.
- [x] Hypergraph reranker without medical knowledge constraints.
- [x] Proposed knowledge-constrained hypergraph reranker.
- [x] Ablation: remove MeSH constraint.
- [x] Ablation: remove PrimeKG relation expansion.
- [x] Ablation: remove biomedical entity overlap.
- [x] Ablation: remove hypergraph diffusion.
- [x] Ablation: replace hypergraph with ordinary pairwise graph.
- [ ] Sensitivity: top-M values 20, 50, 100.
- [ ] Sensitivity: top-k values 1, 3, 5, 10.

## 9. Generation And Faithfulness

- [ ] Add evidence-grounded answer generation only after retrieval is stable.
- [ ] Use top-k retrieved evidence as generation context.
- [ ] Save generated answers under `outputs/generation/`.
- [ ] Evaluate answer accuracy where labels support it.
- [ ] Evaluate citation support rate.
- [ ] Evaluate unsupported claim rate with rule-based or mixed protocol.
- [ ] Evaluate answer-evidence entity consistency.
- [ ] Ensure LLM-as-judge is not the only faithfulness metric.

## 10. Secondary Dataset Robustness

- [ ] Add PubMedQA only after the full primary pipeline works.
- [ ] Add MedMCQA only if the experiment scope remains manageable.
- [ ] Add MedQA-USMLE only if multiple-choice QA generation or selection is stable.
- [ ] Report whether gains are stable across at least two datasets or two splits.

## 11. Literature Verification

- [ ] Verify latest metadata for MedRAG / MIRAGE style medical RAG benchmarking.
- [ ] Verify latest metadata for medical GraphRAG with controlled vocabularies.
- [ ] Verify latest metadata for HyperGraphRAG.
- [ ] Verify latest metadata for cross-granularity hypergraph RAG.
- [ ] Verify latest metadata for self-reflective or iterative medical RAG.
- [ ] Verify latest metadata for knowledge graph enhanced biomedical QA.
- [ ] Keep novelty claims modest: domain-specific constraints, local hypergraph construction, and evidence-oriented evaluation.

## 12. Paper Assets

- [ ] Main retrieval result table.
- [ ] Ablation table.
- [x] Reranking diagnostic table.
- [ ] Top-M and top-k sensitivity table or figure.
- [ ] Case study with interpretable entity or evidence path.
- [ ] Failure analysis.
- [ ] Method diagram.
- [ ] Efficiency analysis.
- [ ] Draft paper structure: Introduction, Related Work, Task Definition, Method, Experimental Setup, Results, Ablation Study, Case Study, Discussion, Conclusion.

## 13. Quality Gates

- [x] All scripts expose CLI arguments for paths and model names.
- [x] No raw data is overwritten.
- [x] No API keys are hard-coded.
- [x] Metrics are written to JSON.
- [x] Predictions are written to JSONL or CSV.
- [x] Deterministic seeds are used where possible.
- [x] Sample test is run before full experiment.
- [x] First-stage recall is checked before reranking.
- [x] No obvious data leakage is found.
- [x] Failure cases are reported honestly if reranking does not improve.
