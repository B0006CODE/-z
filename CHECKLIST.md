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
- [x] Sensitivity: top-k values 1, 3, 5, 10.

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
- [x] Add enhanced first-stage retrieval with fielded BM25, MeSH query expansion, MedCPT dense retrieval, and multi-run RRF fusion.

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
- [x] Case studies showing rescued evidence passages and their MeSH / hypergraph paths.
- [x] Failure analysis for cases where KCH-MedRank loses top-10 evidence.

Current KCH-MedRank BioASQ status:

- Full held-out test results are saved to `results/tables/kch_medrank_bioasq_retrieval.md` and `results/metrics/kch_medrank_bioasq_metrics.json`.
- Full KCH-MedRank improves BioASQ held-out Hybrid RRF MRR@10 from `0.7550` to `0.7664`, Recall@10 / evidence coverage from `0.4636` to `0.4768`, and nDCG@10 from `0.5848` to `0.6013`.
- Paired bootstrap against Hybrid RRF: MRR@10 delta `+0.0114`, p=`0.0050`; Recall@10 delta `+0.0132`, p=`0.0002`; nDCG@10 delta `+0.0165`, p=`0.0002`.
- Hard subset contains 26 held-out questions. Hybrid top10 is intentionally zero on this subset; Full KCH-MedRank recovers Recall@10 / evidence coverage `0.1447`.
- Feature importance is saved to `results/tables/kch_medrank_bioasq_feature_importance.md`. PrimeKG remains low-importance / auxiliary and should not be overclaimed.

Enhanced first-stage retrieval status:

- MeSH descriptor entry terms are extracted from the 2026 MeSH XML into `data/external_knowledge/mesh_synonyms_2026.jsonl`.
- Synonym-aware question MeSH coverage increased to `3823 / 4719` BioASQ questions.
- Fielded BM25 with MeSH expansion improves full-set Recall@100 from original Hybrid `0.6336` to `0.6966`, but has weaker MRR@10 and should be used as a recall source rather than a direct replacement.
- Enhanced four-way RRF raises full-set Recall@100 to `0.7292` with balanced weights and `0.7534` with recall-optimized weights.
- Fusion weights are selected on the validation split (`qid % 5 == 3`), where w122 maximizes Recall@100 (`0.7568`), then reported on held-out test (`0.7388` Recall@100).
- Enhanced KCH-MedRank on the recall-optimized candidate pool improves held-out test MRR@10 to `0.7867`, Recall@10 to `0.5329`, and nDCG@10 to `0.6433`.
- Against true MedCPT cross-encoder reranking on the same held-out test candidate pool, enhanced KCH-MedRank improves Recall@10 from `0.5172` to `0.5329` with paired bootstrap p=`0.0076`; MRR@10 is higher but not significant.
- Flat biomedical knowledge LambdaMART without graph structure is nearly tied with the full model at top-10, so graph and hypergraph features must be framed as interpretable complementary signals.
- Failure analysis on the held-out test split shows `661` rescued gold passages versus `307` lost gold passages at top-10.

## 9. Generation And Faithfulness

- [ ] Add evidence-grounded answer generation only after retrieval is stable.
- [x] Use top-k retrieved evidence as answer-selection context.
- [x] Save PubMedQA answer-selection predictions under `outputs/generation/`.
- [x] Evaluate answer accuracy where labels support it.
- [x] Evaluate citation support rate.
- [x] Evaluate unsupported claim rate with rule-based or mixed protocol.
- [x] Evaluate answer-evidence entity consistency.
- [x] Ensure LLM-as-judge is not the only faithfulness metric.

Current PubMedQA QA diagnostic:

- Lightweight answer selection is implemented in `scripts/run_pubmedqa_qa.py`.
- Summary table is saved to `results/tables/pubmedqa_qa_accuracy.md`.
- Rule-based answer-support diagnostics are implemented in `scripts/evaluate_pubmedqa_faithfulness.py`.
- Faithfulness summary table is saved to `results/tables/pubmedqa_faithfulness.md`.
- PubMedQA test evidence hit@10 is near 1.0 for Dense, Hybrid, and HGB outputs.
- Hybrid and HGB top-10 citation support are both `1.0000`, but supported-answer rate remains bounded by lightweight answer selection (`0.5550` for the majority baseline).
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

- [x] Verify latest metadata for MedRAG / MIRAGE style medical RAG benchmarking.
- [x] Verify latest metadata for medical GraphRAG with controlled vocabularies.
- [x] Verify latest metadata for HyperGraphRAG.
- [x] Verify latest metadata for cross-granularity hypergraph RAG.
- [x] Verify latest metadata for self-reflective or iterative medical RAG.
- [x] Verify latest metadata for knowledge graph enhanced biomedical QA.
- [x] Keep novelty claims modest: domain-specific constraints, local hypergraph construction, and evidence-oriented evaluation.

## 12. Paper Assets

- [x] Main retrieval result table.
- [x] Ablation table.
- [x] Reranking diagnostic table.
- [x] Top-M sensitivity table or figure.
- [x] Top-k sensitivity table or figure.
- [x] Case study with interpretable entity or evidence path.
- [x] Failure analysis.
- [x] Method diagram.
- [x] Efficiency analysis.
- [x] Draft paper structure: Introduction, Related Work, Task Definition, Method, Experimental Setup, Results, Ablation Study, Case Study, Discussion, Conclusion.

## 12A. Manuscript

- [x] Create bilingual LaTeX paper skeleton under `paper/`.
- [x] Create `main_en.tex` and `main_zh.tex`.
- [x] Create matched `sections_en/` and `sections_zh/`.
- [x] Add bilingual synchronization rules to `paper/README.md`.
- [x] Verify literature metadata online before adding citations.
- [x] Create `references.bib` only from verified sources.
- [x] Record verified citation source links.
- [x] Build manuscript result tables.
- [x] Build method diagram.
- [x] Write Introduction in English and Chinese.
- [x] Write Related Work after verifying literature metadata.
- [x] Write Method in English and Chinese.
- [x] Write Experimental Setup in English and Chinese.
- [x] Write Results in English and Chinese.
- [x] Write Ablation Study in English and Chinese.
- [x] Write Case Study and Failure Analysis in English and Chinese.
- [x] Write Discussion and limitations in English and Chinese.
- [x] Write Conclusion in English and Chinese.
- [x] Verify initial `references.bib` entries.
- [x] Compile English PDF.
- [x] Compile Chinese PDF.
- [x] Confirm both language versions are synchronized before submission.

## 12B. English Manuscript Major-Revision Tasks From Internal Review

Internal review decision:

```text
Major Revision before external submission.
```

Core review finding:

- The manuscript is promising and appropriately cautious about clinical claims, but the ablation evidence should frame graph and hypergraph features as complementary rather than dominant.
- The safest current contribution framing is: an interpretable medical evidence control layer for retrieval-augmented generation, implemented as a reproducible biomedical semantic learning-to-rank reranking framework with knowledge-constrained local hypergraph features.
- Do not claim that hyperedges or medical constraints alone drive the performance gains unless additional significance tests support that claim.
- Remove self-undermining language such as `modest claim`, `intended contribution is modest`, and unnecessary `secondary component` wording. The manuscript should remain scientifically cautious without telling reviewers that the contribution is small.
- The paper should connect retrieval improvements to the current LLM-based medical RAG setting by adding a controlled downstream LLM generation loop, without training or fine-tuning a large model.

Required manuscript and analysis tasks:

- [x] Add paired significance tests comparing Full KCH-MedRank against the strongest ablation variants:
  - [x] `Flat biomedical knowledge LambdaMART without graph structure`.
  - [x] `Pairwise graph LTR`.
  - [x] `Remove MeSH hierarchy`.
  - [x] `Remove hypergraph diffusion/centrality`.
- [x] Report confidence intervals and p-values for these ablation comparisons, preferably in `results/metrics/` and `paper/tables/`.
- [x] Clarify metric definitions in the Experimental Setup:
  - [x] whether Recall@k is evidence-level coverage, micro passage recall, macro passage recall, or query-level success.
  - [x] how MRR@k is computed when each question may have multiple gold passages.
  - [x] how nDCG@k handles binary versus graded relevance.
- [x] Add Hit@k or another explicit query-level success metric if MRR is query-level.
- [x] Add an algorithm box or compact pseudocode for:
  - [x] local hypergraph construction.
  - [x] hyperedge weighting and normalization.
  - [x] diffusion propagation.
  - [x] passage-level feature extraction.
  - [x] LambdaMART query-group training and final scoring.
- [x] Specify the complete validation search space for enhanced candidate fusion weights and ranking hyperparameters.
- [x] State explicitly that fusion weight selection and model selection were performed only on validation data.
- [x] Expand reproducibility details:
  - [x] corpus size and passage construction.
  - [x] qrel / gold evidence construction.
  - [x] question-level split rule and leakage checks.
  - [x] model names and hardware where relevant.
- [x] Reframe PubMedQA as a secondary diagnostic unless the same KCH-MedRank configuration is rerun on PubMedQA.
- [x] Revise the title, abstract, introduction, discussion, and conclusion to present KCH-MedRank as an interpretable medical evidence control layer for RAG rather than as a purely traditional IR reranker.
- [x] Delete or rewrite self-deprecating phrases:
  - [x] `modest claim`.
  - [x] `intended contribution is modest`.
  - [x] unnecessary `secondary component` wording.
  - [x] language implying the contribution is small before reviewers judge it.
- [x] Add a controlled downstream LLM generation evaluation if compute permits:
  - [x] choose one fixed open-source instruction LLM, such as Qwen or Llama-family model available locally / reproducibly.
  - [x] use the same prompt template for all retrieval conditions.
  - [ ] compare at least Hybrid RRF + LLM, MedCPT Cross-Encoder + LLM, and KCH-MedRank + LLM.
  - [x] run on PubMedQA first because labels support answer accuracy evaluation.
  - [x] report Accuracy and Macro-F1 where labels allow.
  - [x] report citation support rate, unsupported claim rate, and answer-evidence entity consistency.
  - [x] keep LLM parameters fixed and document model name, decoding settings, prompt, top-k evidence count, and hardware.
  - [x] if answer accuracy does not improve, still report whether evidence support or unsupported-claim metrics improve.
- [x] If downstream LLM generation improves answer accuracy or support metrics, elevate the framing from `retrieval-only reranker` to `white-box evidence selection and control module for medical RAG`.
- [ ] If downstream LLM generation does not improve, keep the main claim focused on evidence retrieval and explicitly discuss this as a limitation rather than hiding the result.
- [x] Update claims in English and Chinese manuscripts together:
  - [x] avoid "hypergraph dominates" or equivalent wording.
  - [x] emphasize significant Recall@10 over MedCPT Cross-Encoder, while describing MRR@10 and nDCG@10 as non-significant positive trends.
  - [x] avoid overclinical claims while still presenting downstream LLM generation as an important RAG-system validation if controlled metrics support it.
- [x] After revision, compile both `paper/main_en.tex` and `paper/main_zh.tex`.
- [x] Review `git status`, stage only appropriate source / manuscript / lightweight result files, commit, and push to `origin main`.

Current 12B revision status:

- Strong-ablation paired bootstrap outputs are saved under `results/metrics/kch_medrank_enhanced_bioasq_bootstrap_vs_*.json` and summarized in `results/tables/kch_medrank_enhanced_strong_ablation_significance.md`.
- Paper tables are saved as `paper/tables/strong_ablation_significance*.tex`.
- PubMedQA Qwen3-8B pilot uses 100 questions, top-5 evidence, temperature 0, fixed JSON citation prompt, and DashScope OpenAI-compatible `qwen3-8b`; metrics are saved to `results/metrics/pubmedqa_qwen3_8b_pilot_metrics.json` and `results/tables/pubmedqa_qwen3_8b_pilot.md`.
- PubMedQA currently compares Hybrid RRF, the available semantic cross-encoder PubMedQA output, and KCH-style hypergraph evidence. A true PubMedQA MedCPT Cross-Encoder condition remains unchecked because the previous local CPU MedCPT smoke test timed out; the manuscript labels this condition honestly as semantic cross-encoder rather than MedCPT.
- Both English and Chinese PDFs compile with bundled Tectonic when `PYTHONUTF8=1` is set. TeX Live is not installed locally; remaining warnings are minor underfull table/case-study boxes and a small candidate-fusion overfull line.

## 12C. Major-Revision Priority Actions After Strict Review

Immediate priority:

- The next work should address reviewer-blocking risks before adding more generation experiments.
- The main risk is an unfair or unclear efficiency claim if KCH-MedRank uses an expensive semantic feature but the feature-generation cost is excluded.
- The second risk is overclaiming hypergraph / medical-knowledge contribution when strong ablations show that `Flat biomedical knowledge LambdaMART without graph structure` is nearly tied with the full model.

Required next experiments and edits:

- [x] Audit every manuscript/table phrase that says `MedCPT Cross-Encoder score`, `cross-encoder score`, `biomedical semantic score`, or `semantic reranker score`.
- [x] Confirm the exact semantic feature used by enhanced KCH-MedRank on BioASQ:
  - [x] if it comes from MedCPT Cross-Encoder, include online CE scoring time in KCH runtime.
  - [x] if it comes from MedCPT dense / dual-encoder retrieval or another precomputed score, state this explicitly and do not describe it as Cross-Encoder scoring.
- [x] Rebuild a fair efficiency table with at least these rows:
  - [x] MedCPT Cross-Encoder, with online CE tokenization and forward scoring.
  - [x] Full KCH-MedRank with the currently used semantic feature, reporting whether the semantic feature is precomputed.
  - [x] KCH-MedRank without Cross-Encoder / without semantic feature.
  - [x] Retrieval-feature-only LambdaMART.
- [x] Report for each efficiency row:
  - [x] whether CE score is used.
  - [x] whether CE is run online.
  - [x] Recall@10.
  - [x] MRR@10.
  - [x] nDCG@10.
  - [x] total reranking time.
  - [x] what costs are excluded, such as first-stage retrieval, model loading, offline training, or precomputed semantic scoring.
- [x] If KCH full depends on online CE scoring, remove the unconditional `19.3x faster` claim and replace it with a conditional statement such as: once semantic features are available, the tabular reranking layer is cheaper than full cross-encoder scoring.
- [x] If no-CE / no-semantic KCH remains competitive with MedCPT CE, report it as the stronger efficiency result.
- [x] Add at least one external retrieval-only dataset or official retrieval split before submission if compute permits:
  - [ ] official BioASQ snippet/document retrieval split, preferred if practical.
  - [x] BEIR biomedical subset such as NFCorpus or TREC-COVID.
  - [ ] PubMedQA retrieval-only diagnostic if the above are too costly.
- [x] Do not expand Qwen3-8B PubMedQA generation until the fair runtime / no-semantic diagnostics are complete.
- [x] Move weak or small PubMedQA generation results to appendix / diagnostic framing unless controlled metrics show clear answer-quality or evidence-support gains.

Required claim edits:

- [x] Reframe the main contribution as a white-box biomedical evidence reranking / evidence-control layer for medical RAG.
- [x] State that enhanced candidate generation, semantic relevance features, and query-group supervised LTR are the dominant measured performance drivers.
- [x] State that hypergraph, MeSH, entity, and PrimeKG relation features provide interpretable complementary signals unless new significance tests show a stronger independent effect.
- [x] Avoid saying or implying that hypergraph learning or medical knowledge constraints are the primary source of the total performance gain.
- [x] Keep PrimeKG as an auxiliary relation source unless normalization and feature analysis show stronger coverage and importance.
- [x] Keep English and Chinese manuscripts synchronized for all claim, limitation, table, and conclusion changes.

Current 12C status:

- Fair efficiency diagnostics are implemented in `scripts/run_efficiency_comparison.py`.
- Full fair efficiency outputs are saved to `results/metrics/efficiency_comparison_bioasq.json`, `results/tables/efficiency_comparison_bioasq.md`, and `paper/tables/reranking_efficiency.tex`.
- The enhanced KCH-MedRank semantic feature source is `outputs/retrieval/medcpt_dense_full_top100.jsonl`; the script classifies it as `dense_or_dual_encoder_predictions`, with `uses_ce_score=false` and `online_ce=false`.
- The previous unconditional `19.3x` speedup wording has been replaced with a fair `18.6x` reranking-stage comparison that states KCH-MedRank does not use Cross-Encoder scores.
- On the held-out BioASQ test split, Full KCH-MedRank reports Recall@10 `0.5329`, MRR@10 `0.7867`, nDCG@10 `0.6433`, and reranking time `37.59s`.
- The no-semantic KCH diagnostic reports Recall@10 `0.5244`, MRR@10 `0.7790`, nDCG@10 `0.6330`, and reranking time `37.50s`.
- Retrieval-feature-only LambdaMART reports Recall@10 `0.5208`, MRR@10 `0.7791`, nDCG@10 `0.6282`, and reranking time `0.84s`.
- MedCPT Cross-Encoder timing is reused from the previous completed CUDA run: Recall@10 `0.5172`, MRR@10 `0.7775`, nDCG@10 `0.6390`, and reranking time `699.46s`.
- English and Chinese PDFs compile with bundled Tectonic after the fair-efficiency update. Remaining warnings are layout/citation rerun warnings already present in the manuscript workflow.
- External BEIR NFCorpus retrieval-only diagnostic is implemented in `scripts/prepare_nfcorpus.py` and `scripts/run_nfcorpus_ltr_diagnostic.py`.
- NFCorpus normalized files are written under `data/processed/`, retrieval predictions under `outputs/retrieval/nfcorpus_*`, and retrieval-feature LambdaMART predictions under `outputs/rerank/nfcorpus_retrieval_ltr_test_top100.jsonl`.
- NFCorpus official split sizes are train `2590`, validation `324`, and test `323` queries. The diagnostic deliberately uses only retrieval features, not project-specific MeSH/entity/PrimeKG/hypergraph features.
- NFCorpus test results are saved to `results/metrics/nfcorpus_retrieval_diagnostic.json` and `results/tables/nfcorpus_retrieval_diagnostic.md`: Retrieval-feature LambdaMART improves Hybrid RRF Recall@10 from `0.1676` to `0.1770`, MRR@10 from `0.5530` to `0.5725`, and nDCG@10 from `0.3436` to `0.3627`.
- NFCorpus paired bootstrap against Hybrid RRF is saved to `results/metrics/nfcorpus_bootstrap_ltr_vs_hybrid.json` and `results/tables/nfcorpus_bootstrap_ltr_vs_hybrid.md`: Recall@10 delta `+0.0093`, p=`0.0034`; nDCG@10 delta `+0.0190`, p=`0.0002`; MRR@10 delta `+0.0195`, p=`0.0592`.
- English and Chinese manuscripts now include NFCorpus as an external retrieval-only robustness diagnostic and move the PubMedQA Qwen3-8B pilot to appendix-style diagnostic framing.
- English and Chinese PDFs compile with bundled Tectonic after the NFCorpus and appendix updates. Remaining warnings are non-blocking underfull/overfull box warnings already consistent with the manuscript workflow.

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
