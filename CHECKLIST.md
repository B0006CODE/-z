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
- [x] Sensitivity: top-M values 20, 50, 100.
- [ ] Sensitivity: top-k values 1, 3, 5, 10.

## 8A. KCH-MedRank Method Upgrade

Working method name:

```text
KCH-MedRank: Knowledge-Constrained Hypergraph Learning-to-Rank with Biomedical Semantic Reranking
```

Rationale:

- Current HGB reranking gives statistically significant but modest BioASQ gains over Hybrid RRF.
- PubMedQA Recall@10 gains are stronger, but MRR is near saturation.
- Dictionary entities and exact PrimeKG matching are too sparse to support a strong medical-knowledge claim.
- The next method version should combine biomedical semantic relevance, MeSH hierarchy-aware constraints, local hypergraph diffusion, and query-group learning-to-rank.

Implementation tasks:

- [ ] Create or update a branch for the method upgrade, preferably `codex/kch-medrank`.
- [x] Add MedCPT or another PubMed-trained biomedical reranking score as a reusable feature.
  - Current full BioASQ run uses `hybrid_metadata.dense_score_fallback` because local CPU MedCPT smoke tests timed out; metrics record this source explicitly and should not be described as full MedCPT evidence.
- [x] Keep a `MedCPT-only` or biomedical-reranker-only baseline.
- [x] Implement MeSH hierarchy loading, including descriptor tree numbers when available.
- [x] Build MeSH hierarchy-aware features:
  - [x] exact descriptor match.
  - [x] parent or ancestor match.
  - [x] sibling match.
  - [x] tree-distance similarity.
  - [x] query concept coverage.
  - [x] passage concept specificity.
  - [x] shared-MeSH candidate cluster features.
- [x] Extend local hypergraph construction with MeSH hierarchy hyperedges.
  - Implemented behind `--enable-mesh-hierarchy-graph-edges`; full CPU run keeps hierarchy-aware features enabled but does not expand all ancestor hyperedges by default for runtime control.
- [x] Keep PrimeKG relation features optional and report coverage before using them in claims.
- [x] Implement LambdaMART / LightGBM ranking under `src/rerank/`.
- [x] Use query-level candidate groups for learning-to-rank.
- [x] Tune ranking hyperparameters on validation only.
- [x] Save KCH-MedRank predictions to `outputs/rerank/`.
- [x] Save KCH-MedRank metrics to `results/metrics/`.
- [x] Save KCH-MedRank summary tables to `results/tables/`.

Required KCH-MedRank comparisons:

- [x] BM25.
- [x] Dense biomedical retriever.
- [x] Hybrid RRF.
- [x] Biomedical semantic reranker only.
- [x] Retrieval-feature-only LambdaMART.
- [x] LambdaMART + biomedical semantic reranker, without hypergraph features.
- [x] Pairwise graph learning-to-rank.
- [x] Hypergraph learning-to-rank without medical knowledge constraints.
- [x] Full KCH-MedRank.

Required KCH-MedRank ablations:

- [x] Remove biomedical semantic reranker score.
- [x] Remove MeSH hierarchy features.
- [x] Remove biomedical entity features.
- [x] Remove hypergraph diffusion and centrality features.
- [x] Remove PrimeKG relation features.
- [x] Replace hypergraph with ordinary pairwise graph.

Required diagnostic evaluations:

- [x] Full held-out test evaluation.
- [x] Hard reranking subset: Hybrid top-100 contains gold evidence, but Hybrid top-10 misses gold evidence.
- [x] Paired bootstrap significance against Hybrid RRF.
- [x] Paired bootstrap significance against biomedical semantic reranker only.
- [x] Feature importance table for the learning-to-rank model.
- [ ] Case studies showing rescued evidence passages and their MeSH / hypergraph paths.
- [ ] Failure analysis for cases where KCH-MedRank loses top-10 evidence.

Current KCH-MedRank BioASQ status:

- Full held-out test results are saved to `results/tables/kch_medrank_bioasq_retrieval.md` and `results/metrics/kch_medrank_bioasq_metrics.json`.
- Full KCH-MedRank improves BioASQ held-out Hybrid RRF MRR@10 from `0.7550` to `0.7664`, Recall@10 / evidence coverage from `0.4636` to `0.4768`, and nDCG@10 from `0.5848` to `0.6013`.
- Paired bootstrap against Hybrid RRF: MRR@10 delta `+0.0114`, p=`0.0050`; Recall@10 delta `+0.0132`, p=`0.0002`; nDCG@10 delta `+0.0165`, p=`0.0002`.
- Hard subset contains 26 held-out questions. Hybrid top10 is intentionally zero on this subset; Full KCH-MedRank recovers Recall@10 / evidence coverage `0.1447`.
- Feature importance is saved to `results/tables/kch_medrank_bioasq_feature_importance.md`. PrimeKG remains low-importance / auxiliary and should not be overclaimed.

## 9. Generation And Faithfulness

- [ ] Add evidence-grounded answer generation only after retrieval is stable.
- [x] Use top-k retrieved evidence as answer-selection context.
- [x] Save PubMedQA answer-selection predictions under `outputs/generation/`.
- [x] Evaluate answer accuracy where labels support it.
- [ ] Evaluate citation support rate.
- [ ] Evaluate unsupported claim rate with rule-based or mixed protocol.
- [ ] Evaluate answer-evidence entity consistency.
- [ ] Ensure LLM-as-judge is not the only faithfulness metric.

Current PubMedQA QA diagnostic:

- Lightweight answer selection is implemented in `scripts/run_pubmedqa_qa.py`.
- Summary table is saved to `results/tables/pubmedqa_qa_accuracy.md`.
- PubMedQA test evidence hit@10 is near 1.0 for Dense, Hybrid, and HGB outputs.
- Simple majority, lexical-rule, and TF-IDF logistic baselines do not yet improve accuracy beyond the majority baseline; this should be reported as a limitation and used to motivate a stronger controlled answer selector before LLM generation.

## 10. Secondary Dataset Robustness

- [x] Add PubMedQA only after the full primary pipeline works.
- [ ] Add MedMCQA only if the experiment scope remains manageable.
- [ ] Add MedQA-USMLE only if multiple-choice QA generation or selection is stable.
- [x] Report whether gains are stable across at least two datasets or two splits.

Current PubMedQA status:

- `qiaojin/PubMedQA` / `pqa_labeled` is normalized into compatible questions, corpus, qrels, answer labels, and MeSH feature files.
- Full PubMedQA first-stage retrieval: Dense MRR@10 `0.9863`, Hybrid RRF MRR@10 `0.9783`.
- PubMedQA HGB held-out reranking improves same-split Hybrid Recall@10 from `0.8200` to `0.8953`, with MRR@10 from `0.9850` to `0.9861`.
- PubMedQA full cross-encoder reranking improves Hybrid MRR@10 from `0.9783` to `0.9806`, but Recall@10 slightly decreases from `0.8297` to `0.8268`.
- BioASQ full held-out cross-encoder timed out on CPU after 2400 seconds; sample10 completed but underperformed Hybrid, so full BioASQ cross-encoder should be rerun on GPU with a biomedical reranker.

## 11. Literature Verification

- [ ] Verify latest metadata for MedRAG / MIRAGE style medical RAG benchmarking.
- [ ] Verify latest metadata for medical GraphRAG with controlled vocabularies.
- [ ] Verify latest metadata for HyperGraphRAG.
- [ ] Verify latest metadata for cross-granularity hypergraph RAG.
- [ ] Verify latest metadata for self-reflective or iterative medical RAG.
- [ ] Verify latest metadata for knowledge graph enhanced biomedical QA.
- [ ] Keep novelty claims modest: domain-specific constraints, local hypergraph construction, and evidence-oriented evaluation.

## 12. Paper Assets

- [x] Main retrieval result table.
- [x] Ablation table.
- [x] Reranking diagnostic table.
- [x] Top-M sensitivity table or figure.
- [ ] Top-k sensitivity table or figure.
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
