# AGENTS.md

## 1. Project Name

English title:

**Knowledge-Constrained Hypergraph Retrieval-Augmented Generation for Evidence-Grounded Medical Question Answering**

Chinese title:

**面向循证医学问答的知识约束超图增强检索生成方法**

This is a new independent research project. Do not assume any existing local project structure, scripts, data files, cached results, or previous experimental conclusions. Build the experiment pipeline from scratch unless the user explicitly provides reusable files later.

The target publication level is SCI Q3/Q4. The project should prioritize feasibility, reproducibility, modest but reliable improvement, and clear medical NLP motivation.

## 2. Research Positioning

This project studies medical evidence-grounded question answering with retrieval-augmented generation. The core hypothesis is:

> Standard medical RAG retrieves evidence mostly by lexical or semantic similarity. It often misses medically meaningful high-order relations among diseases, symptoms, drugs, genes, MeSH concepts, documents, and question entities. A knowledge-constrained hypergraph can model these high-order relations and improve evidence retrieval, answer grounding, and interpretability.

The paper should not claim to invent generic RAG, GraphRAG, or HyperGraphRAG. The contribution is a practical medical-domain method:

> A lightweight, reproducible, knowledge-constrained local hypergraph reranking module for evidence-grounded medical QA.

## 3. Non-Goals

Do not make this project depend on:

- training a large language model,
- fine-tuning a medical LLM,
- expensive multi-GPU training,
- private clinical records,
- manual annotation at large scale,
- image or multimodal VLM experiments,
- full global knowledge graph construction over millions of documents,
- clinical diagnosis or medical advice generation.

Generation may be used only after retrieval is stable. The main scientific claim should be based on evidence retrieval, evidence coverage, faithfulness, and answer accuracy.

## 4. Literature Baseline To Verify

Before writing the final paper or citing exact claims, Codex must verify the latest arXiv / paper metadata. Use these as literature anchors:

- MedRAG / MIRAGE style medical RAG benchmarking.
- Medical Graph RAG using medical controlled vocabularies and trusted sources.
- HyperGraphRAG using hyperedges for n-ary relations in RAG.
- Cross-granularity hypergraph RAG for entity-passage retrieval.
- Self-reflective or iterative medical RAG.
- Knowledge graph enhanced biomedical QA.

Expected positioning:

- MedRAG shows medical RAG is a valid benchmark setting.
- GraphRAG shows structured relations can improve retrieval and reasoning.
- HyperGraphRAG shows hyperedges are useful for high-order relations.
- This project narrows the idea to a feasible medical evidence-grounding system with controlled vocabulary and local hypergraph reranking.

Do not overclaim novelty. State novelty as domain-specific constraint design, local hypergraph construction, and evidence-oriented evaluation.

## 5. Task Definition

Given a medical question `q`, a document corpus `C`, and optional medical knowledge sources `K`, the system should:

1. retrieve candidate evidence passages from `C`;
2. construct a local knowledge-constrained hypergraph around `q` and top-M candidates;
3. rerank candidates using hypergraph structure and medical constraints;
4. generate or select an answer using the top-k evidence passages;
5. report both answer quality and evidence quality.

Formal input:

```text
q: medical question
C: biomedical text corpus
K: medical knowledge source, such as MeSH, UMLS, PrimeKG, or entity dictionaries
```

Formal output:

```text
a: answer
E: ranked supporting evidence passages
P: optional interpretable evidence path or entity chain
```

Primary task:

```text
Evidence retrieval and reranking for medical QA
```

Secondary task:

```text
Evidence-grounded answer generation
```

## 6. Recommended Dataset Strategy

Use public datasets from Hugging Face or official sources. Prefer datasets with clear license, simple loading, and established prior usage.

### 6.1 Primary Dataset

Use one dataset as the main evidence retrieval dataset:

- `rag-datasets/rag-mini-bioasq`

Reason:

- already designed for biomedical RAG / QA;
- includes question-answer and corpus components;
- small enough for reproducible experiments;
- suitable for Recall@k, MRR, and QA evaluation.

### 6.2 Secondary QA Datasets

Use these for robustness:

- `qiaojin/PubMedQA`
- `openlifescienceai/medmcqa`
- `GBaker/MedQA-USMLE-4-options`

Expected usage:

- PubMedQA: evidence-grounded biomedical abstract QA.
- MedMCQA: large-scale medical multiple-choice QA.
- MedQA-USMLE: clinical reasoning style QA.

Do not start with all datasets. Start with one primary dataset, finish the full pipeline, then add secondary datasets.

### 6.3 Knowledge Sources

Recommended knowledge sources:

- MeSH terms from PubMed or MeSH-labeled datasets.
- PrimeKG for disease-drug-gene-phenotype relations.
- UMLS only if access and license are clear.
- Simple biomedical entity dictionaries if UMLS is unavailable.

Preferred first version:

```text
Question / passage entity extraction
  + MeSH concept overlap
  + PrimeKG relation expansion
```

Avoid making UMLS mandatory because licensing and access may create reproducibility problems.

## 7. Project Directory Structure

Use this directory layout:

```text
.
├── AGENTS.md
├── README.md
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external_knowledge/
├── indexes/
│   ├── bm25/
│   └── dense/
├── outputs/
│   ├── retrieval/
│   ├── rerank/
│   └── generation/
├── results/
│   ├── tables/
│   ├── figures/
│   └── metrics/
├── scripts/
├── src/
│   ├── data/
│   ├── retrieval/
│   ├── hypergraph/
│   ├── rerank/
│   ├── generation/
│   └── evaluation/
└── logs/
```

Codex should create files in this structure as needed.

## 8. Experimental Pipeline

The pipeline must be two-stage.

### Stage 1: First-Stage Retrieval

Implement these retrievers:

1. BM25 retriever.
2. Dense retriever using a biomedical or general embedding model.
3. Hybrid retriever using Reciprocal Rank Fusion.

Output top-M candidates per question:

```text
M = 20, 50, 100
```

First-stage recall must be checked before reranking. If the gold evidence is not in top-M, reranking cannot fix the result.

### Stage 2: Knowledge-Constrained Hypergraph Reranking

For each question and its top-M candidates, build a local hypergraph.

Nodes:

- question node,
- candidate passage nodes,
- biomedical entity nodes,
- MeSH concept nodes,
- disease nodes,
- drug nodes,
- gene nodes,
- document nodes.

Hyperedges:

- question-entity hyperedge,
- passage-entity hyperedge,
- document-MeSH hyperedge,
- disease-drug-gene hyperedge from PrimeKG,
- question-passage candidate hyperedge,
- shared-entity hyperedge among candidate passages.

Reranking signals:

- initial retrieval rank,
- dense similarity,
- BM25 score,
- question-passage entity overlap,
- MeSH overlap,
- PrimeKG relation proximity,
- hypergraph diffusion score,
- evidence diversity penalty.

Avoid arbitrary hand-tuned formulas as the only method. If a weighted score is used, weights must be tuned on a validation split and tested in ablation.

Preferred scoring options:

1. Reciprocal Rank Fusion plus hypergraph proximity.
2. Lightweight logistic regression or LambdaMART with interpretable features.
3. Deterministic rank aggregation with validation-selected coefficients.

### Stage 2 Updated Direction: KCH-MedRank

The next method version should be treated as an upgrade of the current local hypergraph reranking pipeline, not as a new project. The working method name is:

```text
KCH-MedRank: Knowledge-Constrained Hypergraph Learning-to-Rank with Biomedical Semantic Reranking
```

The goal is to obtain stronger and more defensible evidence-retrieval gains over strong Hybrid RRF baselines while keeping the paper focused on reproducible biomedical evidence reranking.

Required method changes:

1. Add a biomedical semantic reranking signal, preferably MedCPT or another PubMed-trained biomedical reranker, as an explicit feature rather than as an unreported replacement for the proposed method.
2. Replace pointwise reranking as the main supervised method with listwise or pairwise learning-to-rank, preferably LambdaMART / LightGBM ranking with query-level candidate groups and validation-selected hyperparameters.
3. Extend MeSH features from exact overlap to hierarchy-aware signals, including parent or ancestor match, sibling match, tree-distance similarity, query concept coverage, passage concept specificity, and shared-MeSH candidate clusters.
4. Keep the local hypergraph cross-granularity design: question, passage, document, MeSH concept, biomedical entity, and optional relation nodes connected by query-concept, passage-concept, document-MeSH, shared-concept, hierarchy, and relation hyperedges.
5. Use hypergraph diffusion and centrality as interpretable ranking features, not as an arbitrary hand-tuned standalone formula.
6. Treat PrimeKG as an auxiliary relation feature unless concept normalization improves its coverage. Do not present PrimeKG as the primary source of improvement if validation selects near-zero relation weight.

Required KCH-MedRank baselines and ablations:

- BM25.
- Dense biomedical retriever.
- Hybrid BM25 + Dense RRF.
- MedCPT or biomedical reranker only.
- Retrieval-feature-only LambdaMART.
- LambdaMART + biomedical semantic reranker without hypergraph features.
- Pairwise graph learning-to-rank without hyperedges.
- Hypergraph learning-to-rank without medical knowledge constraints.
- Full KCH-MedRank.
- Remove MedCPT or biomedical semantic reranker.
- Remove MeSH hierarchy features.
- Remove biomedical entity features.
- Remove hypergraph diffusion and centrality features.
- Remove PrimeKG relation features.

Required evaluation additions:

- Full held-out test evaluation remains mandatory.
- Add a hard reranking subset where Hybrid top-100 contains at least one gold evidence passage but Hybrid top-10 misses it. This subset is for diagnostic analysis, not a replacement for the full test set.
- Report Recall@k, MRR@k, nDCG@k, evidence coverage, bootstrap confidence intervals, and p-values against Hybrid RRF and strong semantic reranker baselines.
- Generation remains secondary unless answer accuracy and evidence support improve with controlled metrics.

## 9. Baselines

Minimum baselines:

- No retrieval, direct LLM answer.
- BM25 RAG.
- Dense RAG.
- Hybrid BM25 + Dense RAG.
- Graph reranking without hyperedges.
- Hypergraph reranking without medical knowledge constraints.
- Proposed knowledge-constrained hypergraph reranking.

Optional baselines:

- Query expansion RAG.
- Self-reflective RAG.
- Iterative retrieval RAG.
- Medical entity overlap reranker.

Do not compare only against weak baselines. At minimum, Hybrid BM25 + Dense must be included.

## 10. Ablation Design

Required ablations:

- remove MeSH constraint,
- remove PrimeKG relation expansion,
- remove biomedical entity overlap,
- remove hypergraph diffusion,
- replace hypergraph with ordinary pairwise graph,
- BM25 only,
- dense only,
- hybrid only,
- top-M sensitivity,
- top-k sensitivity.

Recommended top-M values:

```text
20, 50, 100
```

Recommended top-k values:

```text
1, 3, 5, 10
```

The ablation table is essential for SCI Q3/Q4 publication. It should show which medical knowledge constraints actually help.

## 11. Evaluation Metrics

### Retrieval Metrics

Use:

- Recall@k,
- Hit@k,
- MRR,
- nDCG@k if graded relevance exists,
- evidence coverage.

### QA Metrics

Use:

- accuracy for multiple-choice datasets,
- exact match if applicable,
- macro accuracy by question type if labels exist.

### Faithfulness Metrics

If generation is performed, evaluate:

- citation support rate,
- unsupported claim rate,
- answer-evidence entity consistency,
- evidence sufficiency judged by rule-based or LLM-as-judge protocol.

LLM-as-judge must not be the only metric.

## 12. Success Criteria

A result is considered successful if:

- the proposed method improves retrieval MRR or Recall@k over Hybrid BM25 + Dense;
- improvement is stable on at least two datasets or two splits;
- ablation shows medical knowledge constraints contribute;
- first-stage recall is high enough to justify reranking;
- case studies show interpretable entity or evidence paths;
- no obvious data leakage exists.

For SCI Q3/Q4, a modest improvement is acceptable:

```text
Retrieval MRR: +2% to +6%
Recall@5/10: +3% to +10%
QA accuracy: +1% to +4%
Faithfulness or evidence support: +5% or more
```

If QA accuracy does not improve but evidence retrieval and faithfulness improve consistently, the paper can still be positioned as an evidence-grounding method.

## 13. Failure Handling

If reranking does not improve:

1. Check first-stage recall.
2. Check entity extraction quality.
3. Check whether MeSH / PrimeKG mappings are too sparse.
4. Check whether candidate passages are too short or too noisy.
5. Run oracle reranking to estimate upper bound.
6. Report failure honestly and simplify the method.

If first-stage recall is low:

- improve chunking,
- use hybrid retrieval,
- add query expansion,
- increase top-M,
- try biomedical embeddings.

If hypergraph features are too sparse:

- fall back to MeSH overlap,
- use biomedical entity dictionaries,
- use phrase-level entity normalization,
- remove overstrict mapping requirements.

## 14. Implementation Rules For Codex

When Codex works in this project:

- Always inspect the directory before editing.
- Prefer small, testable scripts.
- Keep all outputs versioned.
- Never overwrite raw data.
- Never hard-code API keys.
- Use CLI arguments for paths and model names.
- Write metrics to JSON.
- Write predictions to JSONL or CSV.
- Log dataset size, number of questions, number of corpus passages, and top-M candidate count.
- Add sanity checks before full experiments.
- Use deterministic seeds where possible.

Before a full run, Codex must run a sample test:

```text
10 questions -> retrieve -> rerank -> evaluate -> inspect predictions
```

Only after sample output looks reasonable should Codex run the full dataset.

## 15. Recommended Initial Milestones

Milestone 1: data loading

- Load `rag-datasets/rag-mini-bioasq`.
- Normalize questions, answers, corpus, and evidence labels.
- Save processed files under `data/processed/`.

Milestone 2: first-stage retrieval

- Implement BM25.
- Implement dense retrieval.
- Implement hybrid RRF.
- Report Recall@k and MRR.

Milestone 3: entity and knowledge processing

- Extract biomedical entities from questions and passages.
- Map entities to MeSH or simple normalized terms.
- Load PrimeKG or a small relation table.
- Save entity and relation features.

Milestone 4: local hypergraph reranker

- Build local hypergraph per query over top-M candidates.
- Implement hypergraph proximity / diffusion feature.
- Rerank candidates.

Milestone 5: main experiments

- Compare all baselines.
- Run top-M sensitivity.
- Run ablations.
- Generate main result table.

Milestone 6: answer generation

- Use top-k evidence to generate answers.
- Compare answer accuracy and evidence support.
- Keep generation secondary to retrieval.

Milestone 7: paper assets

- Main result table.
- Ablation table.
- Case study figure.
- Failure analysis.
- Method diagram.

## 16. Preferred Paper Structure

Use this manuscript structure:

1. Introduction
2. Related Work
3. Task Definition
4. Method
5. Experimental Setup
6. Results
7. Ablation Study
8. Case Study
9. Discussion
10. Conclusion

Method section should include:

- first-stage retrieval,
- medical entity extraction,
- knowledge-constrained local hypergraph construction,
- hypergraph reranking,
- evidence-grounded answer generation.

Results section should prioritize:

- retrieval performance,
- evidence quality,
- answer accuracy,
- ablation,
- efficiency.

## 17. SCI Q3/Q4 Positioning

Target article type:

- applied medical NLP,
- biomedical information retrieval,
- healthcare AI,
- medical informatics,
- intelligent systems in medicine.

The paper should emphasize:

- reproducible public datasets,
- clear engineering pipeline,
- interpretable medical constraints,
- stronger evidence retrieval,
- practical computational cost.

Avoid overstating clinical impact. Use phrases such as:

- "supports evidence-grounded medical question answering",
- "improves retrieval of relevant biomedical evidence",
- "enhances interpretability of RAG outputs",
- "may reduce unsupported generation in medical QA settings".

Do not write:

- "diagnoses disease",
- "replaces clinicians",
- "guarantees medical correctness",
- "clinically validated".

## 18. Immediate First Action For Codex

When starting implementation, Codex should first create:

```text
README.md
requirements.txt
configs/default.yaml
scripts/prepare_bioasq.py
scripts/run_bm25.py
scripts/evaluate_retrieval.py
src/
```

Then run the smallest possible pipeline:

```text
prepare data -> BM25 retrieval -> retrieval metrics on sample
```

Do not implement the hypergraph module until the first-stage retrieval baseline is working and measured.

## 19. Version Management

This project is version-managed with GitHub:

```text
Remote repository: https://github.com/B0006CODE/-z.git
Default branch: main
```

When Codex completes a major functional or experimental change, it must:

1. run the relevant sanity checks or tests;
2. review `git status`;
3. stage only appropriate files;
4. commit with a concise, meaningful message;
5. push to `origin main`;
6. report the commit hash and push status to the user.

Tracked files should include:

- source code under `src/`;
- runnable scripts under `scripts/`;
- configuration files under `configs/`;
- documentation and checklists;
- lightweight logs, metrics, and result tables.

Do not commit large reproducible artifacts unless the user explicitly asks for Git LFS handling:

- raw or processed dataset files under `data/`;
- retrieval or rerank indexes under `indexes/`;
- large prediction JSONL files under `outputs/`;
- Python caches or local virtual environments.

The repository contains `.gitignore` rules for these generated artifacts. Preserve those rules unless the user explicitly changes the versioning policy.

## 20. Paper Writing And Bilingual LaTeX Rules

The manuscript is maintained under `paper/` using a generic LaTeX article-style template. Do not bind the manuscript to Elsevier, Springer, IEEE, MDPI, or any other journal-specific template unless the user explicitly requests it later.

Use these paper entry points:

- `paper/main_en.tex` for the English manuscript.
- `paper/main_zh.tex` for the Chinese manuscript.
- `paper/sections_en/` and `paper/sections_zh/` for matched section files.

English and Chinese versions must remain synchronized:

- Each English section file must have a corresponding Chinese section file.
- If one language adds or changes a claim, result, limitation, table reference, citation, or conclusion, update the corresponding location in the other language in the same change.
- If a translation cannot be completed immediately, add an explicit TODO marker in both versions rather than silently changing only one side.

Reference rules:

- Literature metadata must be verified online from reliable sources before citation.
- Do not invent paper titles, authors, years, venues, DOIs, arXiv IDs, PubMed IDs, or BibTeX entries.
- Do not add unverified literature to `paper/references.bib`; keep unverified related work as explicit TODO comments in the manuscript.
- Prefer official sources such as arXiv, PubMed/NCBI, ACL Anthology, IEEE, ACM, Springer, Elsevier, Nature, MDPI, Hugging Face official model pages, and official GitHub repositories for software or dataset information.
- Record verified citation source links in `paper/verified_sources.md`.

Claim and result rules:

- Manuscript claims must match the current experimental evidence.
- The contribution should be framed as an interpretable medical evidence control layer for retrieval-augmented generation, implemented through a reproducible knowledge-constrained local hypergraph learning-to-rank framework for biomedical evidence retrieval and reranking.
- Do not claim to invent RAG, GraphRAG, or HyperGraphRAG.
- Do include a controlled downstream LLM generation loop when feasible, using fixed retrieved evidence and fixed prompts, to show whether better evidence selection improves answer accuracy, citation support, and unsupported-claim behavior. Do not make QA generation the main contribution unless controlled metrics support it.
- Do not present PrimeKG as the primary source of improvement; treat it as auxiliary unless validation and feature analysis support a stronger claim.
- Result tables should preferentially come from `results/tables/` and `results/metrics/`.
- After each major manuscript edit, attempt to compile both English and Chinese PDFs and report any LaTeX environment limitations.

Post-review manuscript revision rules:

- The internal academic-paper-reviewer decision for the current English manuscript is Major Revision before external submission.
- The main review risk is that the current ablation evidence does not prove that knowledge-constrained hypergraph features are the dominant source of improvement. Treat hypergraph, MeSH hierarchy, entity, and relation features as complementary and interpretable unless additional significance tests show stronger evidence.
- Remove self-undermining manuscript language such as `modest claim`, `intended contribution is modest`, and unnecessary `secondary component` wording. Preserve scientific caution by being precise about evidence, not by calling the contribution small.
- Reposition the manuscript away from `traditional IR reranker` framing and toward `white-box / interpretable medical evidence control for RAG`.
- Before strengthening any hypergraph contribution claim, compare Full KCH-MedRank against the strongest ablation variants, especially `LambdaMART + semantic, no hypergraph`, `Pairwise graph LTR`, `Remove MeSH hierarchy`, and `Remove hypergraph diffusion/centrality`, using paired confidence intervals and p-values.
- Clarify metric definitions before submission: state exactly how Recall@k, Hit@k, MRR@k, nDCG@k, and evidence coverage are computed when each question may have multiple gold evidence passages.
- Add or maintain an algorithm box / pseudocode for local hypergraph construction, hyperedge weighting, diffusion, feature extraction, and query-group LambdaMART scoring.
- Document the complete validation search space for enhanced RRF fusion and ranking hyperparameters. Make clear that validation, not test performance, selected weights and model settings.
- PubMedQA should be described as a secondary diagnostic unless the same KCH-MedRank configuration is rerun on PubMedQA. However, PubMedQA is the preferred first dataset for the downstream LLM generation loop because it supports answer-label evaluation.
- For the downstream LLM loop, compare Hybrid RRF + LLM, MedCPT Cross-Encoder + LLM, and KCH-MedRank + LLM under the same prompt template, same top-k evidence count, same model, and same decoding settings. Report Accuracy / Macro-F1 where labels allow, plus citation support rate, unsupported claim rate, and answer-evidence entity consistency.
- English and Chinese manuscripts must be updated together for any claim, metric, limitation, table, or conclusion changed during these revisions.
