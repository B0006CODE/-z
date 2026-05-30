# KCH sample2000 structural diagnostics, 2026-05-30

Scope: BioASQ sample2000, top300 shared-candidate concept expansion, numeric qid order, modulo split with 1,200 train / 400 validation / 400 test queries.

Do not use these diagnostics as a full-test claim. They are for selecting the next structural reranking direction.

## Baseline reference

From `kch_shared_hg_sample2000_top300_v2_composite`:

| Method | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| Hybrid RRF | 0.4485 | 0.8509 | 0.6337 |
| Flat biomedical knowledge LTR | 0.5173 | 0.8719 | 0.6958 |
| Pairwise graph LTR | 0.5180 | 0.8675 | 0.6962 |
| Full KCH-MedRank | 0.5136 | 0.8562 | 0.6885 |

## Structural variants tested

| Variant | Recall@10 | MRR@10 | nDCG@10 | Notes |
| --- | ---: | ---: | ---: | --- |
| Full KCH without global structural counts | 0.5174 | 0.8661 | 0.6952 | Recovers most of the loss from raw global structural counts. |
| Full KCH with capped source reasons | 0.5174 | 0.8642 | 0.6944 | Capping did not improve test ranking. |
| Full KCH rank-gated structural features | 0.5154 | 0.8667 | 0.6925 | Helped hard subset but not full test. |
| Strict-specificity KCH | 0.5146 | 0.8666 | 0.6920 | Linear hyperedge specificity alone was insufficient. |
| Strict-specificity KCH without global structural counts | 0.5171 | 0.8693 | 0.6942 | Best pure structural variant by MRR@10, but not above flat LTR. |
| Strict-specificity KCH rank-gated structural features | 0.5138 | 0.8687 | 0.6919 | Did not improve full test. |
| Strict-specificity sqrt, no global counts | 0.5175 | 0.8653 | 0.6947 | Worse than linear strict on MRR. |

## Validation-selected fusion diagnostic

`run_validation_rrf_fusion.py` selected flat weight 0.55 using validation nDCG@10 for flat biomedical knowledge LTR plus strict-specificity KCH without global structural counts.

| Method | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| Flat biomedical knowledge LTR | 0.5173 | 0.8719 | 0.6958 |
| Strict KCH no global counts | 0.5171 | 0.8693 | 0.6942 |
| Validation RRF fusion, flat weight 0.55 | 0.5201 | 0.8726 | 0.6972 |

Bootstrap, 5,000 resamples:

| Comparison | Metric@10 | Delta | 95% CI | p |
| --- | --- | ---: | ---: | ---: |
| Fusion vs Hybrid | Recall | +0.0715 | [+0.0578, +0.0860] | 0.0004 |
| Fusion vs Hybrid | MRR | +0.0217 | [+0.0024, +0.0411] | 0.0276 |
| Fusion vs Hybrid | nDCG | +0.0635 | [+0.0511, +0.0757] | 0.0004 |
| Fusion vs Flat | Recall | +0.0028 | [-0.0003, +0.0065] | 0.0832 |
| Fusion vs Flat | MRR | +0.0008 | [-0.0006, +0.0022] | 0.3083 |
| Fusion vs Flat | nDCG | +0.0013 | [-0.0007, +0.0033] | 0.1852 |
| Fusion vs Pairwise | Recall | +0.0020 | [-0.0017, +0.0057] | 0.3103 |
| Fusion vs Pairwise | MRR | +0.0051 | [-0.0032, +0.0133] | 0.2304 |
| Fusion vs Pairwise | nDCG | +0.0010 | [-0.0033, +0.0050] | 0.6295 |

## Interpretation

The best validation-selected fusion numerically exceeds flat and pairwise on all three top-10 metrics, but none of those gains are significant on sample2000. The defensible current claim remains: shared-cluster expansion plus supervised KCH-style reranking significantly improves over Hybrid RRF, while hypergraph-specific and strict-structural signals are complementary but not yet proven dominant over flat or pairwise knowledge LTR.

Next step: avoid claiming structural dominance. If more work is desired, evaluate the validation-selected fusion and strict-no-global variant on a larger held-out/full split, or add an external retrieval dataset before strengthening the hypergraph contribution claim.
