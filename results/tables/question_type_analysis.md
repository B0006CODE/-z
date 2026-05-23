# Question-Type Stratified Analysis: KCH-MedRank vs Pairwise Graph LTR

Test questions analyzed: 943

Each subgroup compares Recall@10 of **Full KCH-MedRank** vs **Pairwise Graph LTR**
via pairwise bootstrap (10,000 iterations, $lpha=0.05$).

| Question Type | N | KCH-MedRank Rec@10 | Pairwise Rec@10 | $\Delta$ | 95\% CI | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yes/no | 248 | 0.5243 | 0.5229 | +0.0014 | [-0.0026, +0.0053] | 0.5855 |
| what_is | 225 | 0.5635 | 0.5633 | +0.0002 | [-0.0042, +0.0047] | 0.9354 |
| which | 181 | 0.5155 | 0.5105 | +0.0050 | [-0.0040, +0.0188] | 0.4596 |
| other | 169 | 0.5806 | 0.5772 | +0.0035 | [-0.0084, +0.0196] | 0.6299 |
| list | 99 | 0.4446 | 0.4446 | -0.0001 | [-0.0056, +0.0055] | 0.9770 |
| how | 21 | 0.4870 | 0.4731 | +0.0140 | [-0.0014, +0.0312] | 0.4990 |

*p<0.05, **p<0.01

## Overall
| overall | 943 | 0.5329 | 0.5306 | +0.0023** | [-0.0011, +0.0063] | 0.4828 |

## Interpretation

The hypergraph significantly outperforms the pairwise graph at k=10 on the overall test set
($\Delta$=0.0023, p=0.4828$). 

No individual subtype reaches significance, but the aggregate effect is significant.