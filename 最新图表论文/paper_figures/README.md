# KCH-MedRank Publication Figures

This folder builds publication-quality data figures for the English KCH-MedRank manuscript without editing the original paper files.

## Install

From the repository root:

```powershell
python -m pip install -r .\最新图表论文\paper_figures\requirements.txt
```

## Regenerate All Figures

```powershell
python .\最新图表论文\paper_figures\scripts\make_all.py
```

`make_all.py` runs table extraction first and then regenerates every figure. Plotting scripts read only CSV files from `paper_figures/data/processed/`.

## Outputs

Figures are exported in three formats:

- `paper_figures/figures/pdf/`
- `paper_figures/figures/svg/`
- `paper_figures/figures/png/`

Expected figure names:

- `fig2_main_results`
- `fig3_recall_at_k`
- `fig4_structural_ablation`
- `fig5_subgroup_forest`
- `fig6_efficiency_pareto`
- `fig7_interpretability`
- `fig8_synonym_normalization`

## Data Sources

Each processed CSV contains `source_table` and `notes` columns.

| Figure | Processed CSV | Source |
| --- | --- | --- |
| Figure 2 | `data/processed/fig2_main_results.csv` | `tab:enhanced-bioasq-main-results`, `tab:enhanced-bioasq-bootstrap-hybrid`, `tab:enhanced-bioasq-significance` |
| Figure 3 | `data/processed/fig3_recall_at_k.csv` | Held-out BioASQ metric JSON logs under `results/metrics/`; k=10 and k=100 align with the English main-result table where reported |
| Figure 4 | `data/processed/fig4_structural_ablation.csv` | `tab:kch-ablation-summary` |
| Figure 5 | `data/processed/fig5_subgroup_forest.csv` | `tab:hypergraph-vs-pairwise-stratified` |
| Figure 6 | `data/processed/fig6_efficiency_pareto.csv` | `tab:reranking-efficiency` |
| Figure 7 | `data/processed/fig7_interpretability.csv` | `tab:interpretability-mechanisms`, `tab:mech-cooccurrence` |
| Figure 8 | `data/processed/fig8_synonym_normalization.csv` | `tab:normalization-ablation`, `tab:feature-separability` |

Raw source snapshots are copied into:

- `data/raw/source_tables/`
- `data/raw/source_metrics/`
- `data/raw/source_manifest.csv`

## Missing Data And Assumptions

- Figure 2 Panel C reports KCH-MedRank minus MedCPT Cross-Encoder point estimates and p-values. The manuscript table does not report 95% confidence intervals for this comparison, so `ci_lower` and `ci_upper` are intentionally blank in the CSV.
- Figure 3 uses metric JSON logs for Recall@k at k = 1, 3, 5, 10, 20, 50, 100. No intermediate recall points are inferred.
- No values are manually invented. When a source table lacks a value, the processed CSV leaves it blank and records the reason in `notes`.
- Figure 7 uses `upsetplot.from_memberships` to keep mechanism co-occurrence encoded as set intersections, then renders a compact UpSet-style matrix for layout control.

## Figure Messages

- Figure 2: KCH-MedRank improves top-10 retrieval over Enhanced Hybrid RRF and significantly improves Recall@10 over MedCPT Cross-Encoder.
- Figure 3: KCH-MedRank improves early-rank recall while sharing the same enhanced top-100 candidate ceiling as the semantic and hybrid rerankers.
- Figure 4: Flat biomedical knowledge features explain most aggregate gain; the full hypergraph model gives smaller incremental changes at top-10 depth.
- Figure 5: Hypergraph benefit is strongest at Recall@5 for indirect-evidence settings, especially when direct MeSH overlap is absent.
- Figure 6: KCH-MedRank preserves or improves Recall@10 while avoiding online cross-encoder latency under cached-feature reranking.
- Figure 7: Rescued gold passages are dominated by MeSH hierarchy paths, with frequent concurrent mechanisms.
- Figure 8: Synonym-aware normalization features give small consistent validation gains and show clear gold/non-gold feature separability.

## Validation

`data/processed/validation_report.txt` records consistency checks between the main result table, comparison tables, structural ablation values, and interpretability totals.

