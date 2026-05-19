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

The default dense model is `abhinand/MedEmbed-small-v0.1`, chosen as a lightweight medical retrieval baseline. Replace it with another sentence-transformers compatible model by passing `--model-name`.

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

Current diagnostic result: validation selected small weights (`hypergraph_weight=0.02`, `entity_weight=0.02`), but test MRR@10 and nDCG@10 are slightly below Hybrid RRF. Treat this as a failure signal, not a positive result. The likely causes are sparse question-entity coverage, weak dictionary-only MeSH approximation, and no PrimeKG relation expansion yet.

Additional ablations:

```powershell
python scripts/run_hypergraph_rerank.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl --output outputs/rerank/no_knowledge_hypergraph_test_top100.jsonl --metrics-output results/metrics/no_knowledge_hypergraph_test_top100_metrics.json --top-k 100 --structure no_knowledge_hypergraph --tune-weights --target-split test --entity-grid 0
python scripts/run_hypergraph_rerank.py --config configs/default.yaml --predictions outputs/retrieval/hybrid_full_top100.jsonl --output outputs/rerank/pairwise_graph_test_top100.jsonl --metrics-output results/metrics/pairwise_graph_test_top100_metrics.json --top-k 100 --structure pairwise_graph --tune-weights --target-split test
```

The ablation summary is saved to `results/tables/ablation_retrieval.md`. It shows that the current best held-out behavior is essentially Hybrid RRF plus a tiny entity-coverage term. The next technical priority is better medical concept mapping and relation expansion, not generation.
