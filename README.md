# Knowledge-Constrained Hypergraph RAG for Medical QA

This repository implements a reproducible research pipeline for evidence-grounded biomedical question answering.

The project starts with a conservative baseline:

1. prepare `rag-datasets/rag-mini-bioasq`;
2. run first-stage BM25 retrieval;
3. evaluate retrieval quality on a 10-question sanity sample;
4. expand only after the first-stage retrieval baseline is measured.

The later method will add dense retrieval, hybrid retrieval, biomedical entity processing, knowledge-constrained local hypergraph reranking, ablations, and optional evidence-grounded generation.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## First Sanity Run

```powershell
python scripts/prepare_bioasq.py --config configs/default.yaml --sample-size 10
python scripts/run_bm25.py --config configs/default.yaml --sample-limit 10 --top-k 100
python scripts/evaluate_retrieval.py --config configs/default.yaml --ks 1 3 5 10 20 50 100 --only-predicted-qids
```

## Dense And Hybrid Retrieval

The default dense model is `abhinand/MedEmbed-small-v0.1`, chosen as a lightweight medical retrieval baseline. Replace it with another sentence-transformers compatible model by passing `--model-name`. Dense retrieval and cross-encoder reranking use `device: auto` from `configs/default.yaml`, which resolves to CUDA when a compatible GPU is available and otherwise falls back to CPU. Pass `--device cpu` only for CPU-only timing or debugging.

```powershell
python scripts/run_dense.py --config configs/default.yaml --sample-limit 10 --top-k 100
python scripts/evaluate_retrieval.py --config configs/default.yaml --predictions outputs/retrieval/dense_sample.jsonl --output results/metrics/dense_sample_metrics.json --ks 1 3 5 10 20 50 100 --only-predicted-qids
python scripts/run_hybrid.py --config configs/default.yaml
python scripts/evaluate_retrieval.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_sample.jsonl --output results/metrics/hybrid_sample_metrics.json --ks 1 3 5 10 20 50 100 --only-predicted-qids
```

Full first-stage retrieval:

```powershell
python scripts/run_bm25.py --config configs/default.yaml --output outputs/retrieval/bm25_full_top100.jsonl --top-k 100
python scripts/run_dense.py --config configs/default.yaml --output outputs/retrieval/dense_full_top100.jsonl --top-k 100
python scripts/run_hybrid.py --config configs/default.yaml --bm25-predictions outputs/retrieval/bm25_full_top100.jsonl --dense-predictions outputs/retrieval/dense_full_top100.jsonl --output outputs/retrieval/hybrid_full_top100.jsonl --top-k 100
python scripts/evaluate_retrieval.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl --output results/metrics/hybrid_full_top100_metrics.json --ks 1 3 5 10 20 50 100
python scripts/summarize_retrieval.py
```

Outputs are written to:

- processed data: `data/processed/`
- retrieval predictions: `outputs/retrieval/`
- metrics: `results/metrics/`
- logs: `logs/`

## Version Management

The Git repository tracks source code, configs, documentation, logs, metrics, and result tables. Reproducible generated artifacts under `data/`, `indexes/`, and large prediction JSONL files under `outputs/` are ignored to keep GitHub pushes stable and below repository limits. Regenerate them with the commands above when needed.

## Current Scope

The current implementation covers data preparation, BM25 retrieval, dense retrieval, hybrid RRF retrieval, and retrieval evaluation. Hypergraph reranking should be implemented only after the first-stage retrieval baselines are measured and checked.

Current full-dataset first-stage result table is saved to `results/tables/first_stage_retrieval.md`.

## Entity Features

The first entity-processing version uses a local dictionary derived from the public BioASQ corpus and question terms. It avoids UMLS licensing requirements and records sparsity before hypergraph reranking.

```powershell
python scripts/build_entity_dictionary.py --config configs/default.yaml
python scripts/extract_entities.py --config configs/default.yaml
python scripts/analyze_entity_overlap.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl
python scripts/run_entity_overlap_rerank.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl
python scripts/evaluate_retrieval.py --config configs/default.yaml --predictions outputs/rerank/entity_overlap_full_top100.jsonl --output results/metrics/entity_overlap_full_top100_metrics.json --ks 1 3 5 10 20 50 100
```

## Local Hypergraph Reranking

The first local hypergraph reranker builds a query-specific hypergraph over each question, top-M candidate passages, dictionary entity nodes, entity-type concept nodes, question-passage candidate edges, passage-entity edges, and shared-entity hyperedges. It then computes a diffusion feature and fuses it with the original Hybrid RRF rank.

Use validation tuning and evaluate on the deterministic test split:

```powershell
python scripts/run_hypergraph_rerank.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl --output outputs/rerank/hypergraph_test_top100.jsonl --metrics-output results/metrics/hypergraph_test_top100_metrics.json --top-k 100 --tune-weights --target-split test
python scripts/evaluate_retrieval.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_test_top100.jsonl --output results/metrics/hybrid_test_top100_metrics.json --ks 1 3 5 10 20 50 100 --only-predicted-qids
```

Current diagnostic result: validation selected small weights (`hypergraph_weight=0.02`, `entity_weight=0.02`), but test MRR@10 and nDCG@10 are slightly below Hybrid RRF. Treat this as a failure signal, not a positive result. The likely causes are sparse question-entity coverage, weak lexical concept mapping, and no PrimeKG relation expansion yet.

Additional ablations:

```powershell
python scripts/run_hypergraph_rerank.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl --output outputs/rerank/no_knowledge_hypergraph_test_top100.jsonl --metrics-output results/metrics/no_knowledge_hypergraph_test_top100_metrics.json --top-k 100 --structure no_knowledge_hypergraph --tune-weights --target-split test --entity-grid 0
python scripts/run_hypergraph_rerank.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl --output outputs/rerank/pairwise_graph_test_top100.jsonl --metrics-output results/metrics/pairwise_graph_test_top100_metrics.json --top-k 100 --structure pairwise_graph --tune-weights --target-split test
```

The ablation summary is saved to `results/tables/ablation_retrieval.md`. It shows that the current best held-out behavior is essentially Hybrid RRF plus a tiny entity-coverage term. The next technical priority is better medical concept mapping and relation expansion, not generation.

## Diagnostics And PubMed MeSH

Reranking diagnostics show substantial oracle headroom inside Hybrid top100, but weak dictionary entity separability:

```powershell
python scripts/analyze_rerank_diagnostics.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl --output results/metrics/rerank_diagnostics_hybrid_top100.json --top-m 100
```

PubMed MeSH metadata can be fetched reproducibly from NCBI E-utilities using passage ids as PMIDs. The script resumes from existing output and does not require an API key.

```powershell
python scripts/fetch_pubmed_mesh.py --config configs/default.yaml --batch-size 200 --sleep 0.5
python scripts/build_mesh_features.py --config configs/default.yaml
python scripts/analyze_mesh_overlap.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl --output results/metrics/mesh_overlap_stats.json --top-m 100
python scripts/run_mesh_overlap_rerank.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl --output outputs/rerank/mesh_overlap_test_top100.jsonl --metrics-output results/metrics/mesh_overlap_test_top100_metrics.json --top-k 100 --tune-weights --target-split test
python scripts/run_hypergraph_rerank.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl --output outputs/rerank/hypergraph_mesh_test_top100.jsonl --metrics-output results/metrics/hypergraph_mesh_test_top100_metrics.json --top-k 100 --tune-weights --target-split test
python scripts/run_hypergraph_rerank.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl --output outputs/rerank/no_mesh_hypergraph_test_top100.jsonl --metrics-output results/metrics/no_mesh_hypergraph_test_top100_metrics.json --top-k 100 --tune-weights --target-split test --disable-mesh
```

Current MeSH status:

- PubMed MeSH fetched for all 40221 corpus passages.
- 37356 passages have non-generic MeSH terms.
- 3375 / 4719 questions have lexical MeSH matches.
- MeSH overlap AUC is about 0.621, better than dictionary entity overlap AUC about 0.576.
- Simple MeSH overlap reranking gives only tiny held-out changes.
- Adding MeSH as question-MeSH and document-MeSH hyperedges selects `mesh_weight=0.02`, but still does not beat Hybrid RRF on held-out MRR@10 or nDCG@10.

## PrimeKG Relation Expansion

PrimeKG is used as an optional external knowledge source. The project streams the public CSV from Harvard Dataverse and filters it down to relation rows whose endpoint names exactly match the local entity dictionary. The full PrimeKG CSV is not committed.

```powershell
python scripts/build_primekg_relations.py --config configs/default.yaml
python scripts/analyze_primekg_relations.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl --output results/metrics/primekg_relation_stats.json --top-m 100
python scripts/run_hypergraph_rerank.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl --output outputs/rerank/hypergraph_primekg_test_top100.jsonl --metrics-output results/metrics/hypergraph_primekg_test_top100_metrics.json --top-k 100 --tune-weights --target-split test --hypergraph-grid 0,0.02,0.05 --entity-grid 0,0.02,0.05 --mesh-grid 0,0.02,0.05 --relation-grid 0,0.02,0.05
```

Current PrimeKG status:

- The stream scan produced 5766 filtered project-local relation rows.
- The filtered relations cover 1385 local entity ids.
- Only 330 / 4719 questions have any PrimeKG relation among Hybrid top100 candidates.
- PrimeKG relation-count AUC is about 0.509, so validation selects `relation_weight=0.0`.
- This is a useful negative result: exact-name relation expansion is too sparse; better concept normalization is needed before PrimeKG can support a stronger method claim.

## PubMedQA And Cross-Encoder Reranking

PubMedQA is added as the second dataset using `qiaojin/PubMedQA` / `pqa_labeled`. Each abstract section is treated as one corpus passage, all sections from the question's source abstract are treated as relevant evidence, and yes/no/maybe labels are saved for later QA accuracy evaluation.

```powershell
python scripts/prepare_pubmedqa.py --config configs/default.yaml --output-prefix data/processed/pubmedqa_pqa_labeled
python scripts/run_bm25.py --config configs/default.yaml --corpus data/processed/pubmedqa_pqa_labeled_corpus.jsonl --questions data/processed/pubmedqa_pqa_labeled_questions.jsonl --output outputs/retrieval/pubmedqa_bm25_full_top100.jsonl --index-path indexes/bm25/pubmedqa_bm25.pkl --top-k 100 --rebuild-index
python scripts/run_dense.py --config configs/default.yaml --corpus data/processed/pubmedqa_pqa_labeled_corpus.jsonl --questions data/processed/pubmedqa_pqa_labeled_questions.jsonl --output outputs/retrieval/pubmedqa_dense_full_top100.jsonl --index-path indexes/dense/pubmedqa_medembed_small.npz --top-k 100 --rebuild-index
python scripts/run_hybrid.py --config configs/default.yaml --bm25-predictions outputs/retrieval/pubmedqa_bm25_full_top100.jsonl --dense-predictions outputs/retrieval/pubmedqa_dense_full_top100.jsonl --output outputs/retrieval/pubmedqa_hybrid_full_top100.jsonl --top-k 100
python scripts/run_cross_encoder_rerank.py --config configs/default.yaml --questions data/processed/pubmedqa_pqa_labeled_questions.jsonl --corpus data/processed/pubmedqa_pqa_labeled_corpus.jsonl --qrels data/processed/pubmedqa_pqa_labeled_qrels.jsonl --predictions outputs/retrieval/pubmedqa_hybrid_full_top100.jsonl --output outputs/rerank/pubmedqa_cross_encoder_full_top100.jsonl --metrics-output results/metrics/pubmedqa_cross_encoder_full_top100_metrics.json --top-m 100 --top-k 100 --max-length 256 --batch-size 64
```

Current PubMedQA retrieval summary is saved to `results/tables/pubmedqa_retrieval.md`. Dense retrieval is the strongest first-stage baseline on this dataset. Cross-encoder reranking improves Hybrid RRF MRR@10 from `0.9783` to `0.9806`, but slightly reduces Recall@10 from `0.8297` to `0.8268`.

The HGB knowledge reranker can be reused on PubMedQA after building PubMedQA-specific entity files:

```powershell
python scripts/build_entity_dictionary.py --config configs/default.yaml --corpus data/processed/pubmedqa_pqa_labeled_corpus.jsonl --questions data/processed/pubmedqa_pqa_labeled_questions.jsonl --output data/processed/pubmedqa_entity_dictionary.jsonl --min-count 2
python scripts/extract_entities.py --config configs/default.yaml --dictionary data/processed/pubmedqa_entity_dictionary.jsonl --questions data/processed/pubmedqa_pqa_labeled_questions.jsonl --corpus data/processed/pubmedqa_pqa_labeled_corpus.jsonl --question-output data/processed/pubmedqa_question_entities.jsonl --passage-output data/processed/pubmedqa_passage_entities.jsonl --stats-output results/metrics/pubmedqa_entity_feature_stats.json
python scripts/run_learning_rerank.py --config configs/default.yaml --predictions outputs/retrieval/pubmedqa_hybrid_full_top100.jsonl --qrels data/processed/pubmedqa_pqa_labeled_qrels.jsonl --question-entities data/processed/pubmedqa_question_entities.jsonl --passage-entities data/processed/pubmedqa_passage_entities.jsonl --question-mesh data/processed/pubmedqa_pqa_labeled_question_mesh.jsonl --passage-mesh data/processed/pubmedqa_pqa_labeled_passage_mesh.jsonl --relations data/external_knowledge/pubmedqa_primekg_relations_missing.jsonl --output outputs/rerank/pubmedqa_learning_hgb_test_top100.jsonl --metrics-output results/metrics/pubmedqa_learning_hgb_test_top100_metrics.json --model hist_gradient --feature-set all --top-k 100
```

Current PubMedQA HGB held-out result is saved to `results/tables/pubmedqa_hgb_test.md`: same-split Hybrid RRF MRR@10 `0.9850` vs HGB `0.9861`, and Recall@10 `0.8200` vs `0.8953`. This supports the evidence-coverage argument more strongly than a pure MRR argument.

BioASQ full held-out cross-encoder scoring with `cross-encoder/ms-marco-MiniLM-L-6-v2` timed out on CPU after 2400 seconds. The BioASQ 10-question sanity run completed but underperformed Hybrid RRF. Treat this as a warning that a generic MS MARCO cross-encoder is not enough for the biomedical setting; use a biomedical cross-encoder and GPU for the full BioASQ reranking comparison.

Biomedical reranker candidates are summarized in `results/tables/biomedical_reranker_candidates.md`. The cross-encoder script now also supports Hugging Face `AutoModelForSequenceClassification` rerankers:

```powershell
python scripts/run_cross_encoder_rerank.py --config configs/default.yaml --questions data/processed/pubmedqa_pqa_labeled_questions.jsonl --corpus data/processed/pubmedqa_pqa_labeled_corpus.jsonl --qrels data/processed/pubmedqa_pqa_labeled_qrels.jsonl --predictions outputs/retrieval/pubmedqa_hybrid_full_top100.jsonl --output outputs/rerank/pubmedqa_medcpt_cross_encoder_sample2_top5.jsonl --metrics-output results/metrics/pubmedqa_medcpt_cross_encoder_sample2_top5_metrics.json --model-name ncbi/MedCPT-Cross-Encoder --backend transformers_sequence_classification --top-m 5 --top-k 5 --sample-limit 2 --batch-size 4 --max-length 256 --device auto --only-predicted-qids
```

Current CPU status: this MedCPT smoke test did not finish within 300 seconds in the local environment, before writing metrics. Biomedical cross-encoder experiments should use GPU whenever available; keep CPU runs to tiny timing tests only.

## PubMedQA Answer Selection

PubMedQA yes/no/maybe answer-selection baselines are intentionally lightweight and do not use LLM generation. They consume top-k evidence from retrieval or reranking outputs and report accuracy, macro-F1, per-label F1, and evidence hit@k.

```powershell
python scripts/run_pubmedqa_qa.py --config configs/default.yaml --predictions outputs/retrieval/pubmedqa_hybrid_full_top100.jsonl --output outputs/generation/pubmedqa_hybrid_qa_test.jsonl --metrics-output results/metrics/pubmedqa_hybrid_qa_test_metrics.json --table-output results/tables/pubmedqa_hybrid_qa_accuracy.md --top-ks 1 3 5 10
python scripts/run_pubmedqa_qa.py --config configs/default.yaml --predictions outputs/rerank/pubmedqa_learning_hgb_test_top100.jsonl --train-predictions outputs/retrieval/pubmedqa_hybrid_full_top100.jsonl --output outputs/generation/pubmedqa_hgb_qa_test.jsonl --metrics-output results/metrics/pubmedqa_hgb_qa_test_metrics.json --table-output results/tables/pubmedqa_hgb_qa_accuracy.md --top-ks 1 3 5 10
python scripts/summarize_pubmedqa_qa.py --metrics results/metrics/pubmedqa_dense_qa_test_metrics.json results/metrics/pubmedqa_hybrid_qa_test_metrics.json results/metrics/pubmedqa_hgb_qa_test_metrics.json --output results/tables/pubmedqa_qa_accuracy.md
```

Current QA diagnostic: evidence hit@10 is near 1.0 on PubMedQA test, but the simple majority, lexical-rule, and TF-IDF logistic baselines do not produce a strong answer-accuracy gain. This is useful negative evidence: retrieval coverage is high, but yes/no/maybe decision-making needs a stronger controlled answer selector before LLM generation is introduced.
