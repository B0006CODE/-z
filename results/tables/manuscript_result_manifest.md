# Manuscript Result Manifest

This file identifies which result artifacts are used by the current English and Chinese manuscripts. It also separates deprecated diagnostics from paper baselines so that old experiment rows are not confused with reported results.

## Main BioASQ Retrieval Results

| Manuscript method name | Source artifact | Notes |
| --- | --- | --- |
| Original Hybrid RRF | `results/metrics/hybrid_test_top100_metrics.json` and manuscript table copy | Original BM25 + dense RRF baseline. |
| Enhanced Hybrid w122 | `results/metrics/enhanced_hybrid_w122_test_top100_metrics.json` and `paper/tables/enhanced_bioasq_main_results.tex` | Validation-selected enhanced candidate ordering. |
| MedCPT Cross-Encoder | `results/metrics/medcpt_cross_encoder_enhanced_bioasq_test_top100_metrics.json` | True MedCPT Cross-Encoder baseline on the enhanced top-100 candidate pool. |
| Enhanced KCH-MedRank | `results/metrics/kch_medrank_enhanced_bioasq_full_kch_medrank_metrics.json` | Full reported KCH-MedRank result. |

## Main Significance And Ablation Artifacts

| Manuscript table | Source artifact | Purpose |
| --- | --- | --- |
| `paper/tables/enhanced_bioasq_bootstrap_vs_hybrid.tex` | paired bootstrap output copied from enhanced BioASQ logs | Tests Full KCH-MedRank against enhanced Hybrid RRF. |
| `paper/tables/enhanced_bioasq_significance.tex` | paired bootstrap output copied from enhanced BioASQ logs | Tests Full KCH-MedRank against true MedCPT Cross-Encoder. |
| `paper/tables/kch_ablation_summary.tex` | `results/tables/kch_medrank_enhanced_bioasq_retrieval.md` plus corrected MedCPT table row | Summarizes strong and leave-one-group ablations. |
| `paper/tables/strong_ablation_significance.tex` | paired bootstrap output copied from enhanced BioASQ logs | Tests Full KCH-MedRank against the strongest ablation variants. |
| `paper/tables/strong_ablation_interpretation.tex` | derived interpretation table, no new metrics | Clarifies what the close ablation comparisons do and do not support. |
| `paper/tables/reranking_efficiency.tex` | `results/metrics/efficiency_comparison_bioasq.json` and `results/tables/efficiency_comparison_bioasq.md` | Reports reranking-stage wall-clock timing against MedCPT Cross-Encoder on the same enhanced top-100 held-out candidate pool. |

## Deprecated Or Diagnostic Rows

| Artifact row | Status | Reason |
| --- | --- | --- |
| `Deprecated semantic-score-only ordering diagnostic (not the MedCPT Cross-Encoder baseline)` in `results/tables/kch_medrank_enhanced_bioasq_retrieval.md` | Deprecated diagnostic, not a manuscript baseline | This row sorts candidates by an earlier semantic score field and produces extremely low top-10 ranking metrics. It is not the true MedCPT Cross-Encoder baseline used in the paper. |

## PubMedQA Diagnostics

| Manuscript method name | Source artifact | Notes |
| --- | --- | --- |
| PubMedQA retrieval diagnostic | `paper/tables/pubmedqa_retrieval_diagnostic.tex` | Secondary evidence-coverage diagnostic, not the main dataset result. |
| PubMedQA Qwen3-8B pilot | `paper/tables/pubmedqa_qwen_pilot.tex` | 100-question controlled generation pilot; not used as a central generation claim. |
