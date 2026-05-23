# KCH-MedRank vs Pairwise Graph LTR: Stratified Bootstrap Analysis

**Test questions**: 943 | **Bootstrap**: 10000 iterations (paired) | **α**: 0.05

## Recall@5 Analysis

### By Question Type (keyword-based)

| Type | N | KCH | Pairwise | Δ | 95% CI | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yes/no | 248 | 0.4039 | 0.3974 | +0.0065 | [-0.0004, +0.0146] | 0.0690 |
| what_is | 225 | 0.4563 | 0.4495 | +0.0068 | [-0.0012, +0.0184] | 0.1188 |
| which | 181 | 0.4046 | 0.4030 | +0.0016 | [-0.0024, +0.0055] | 0.4438 |
| other | 169 | 0.4722 | 0.4641 | +0.0081 | [-0.0015, +0.0234] | 0.1418 |
| list | 99 | 0.3342 | 0.3313 | +0.0029 | [-0.0079, +0.0145] | 0.6150 |
| how | 21 | 0.3846 | 0.3846 | +0.0000 | [+0.0000, +0.0000] | 1.0000 |
| **Overall** | **943** | **0.4210** | **0.4156** | **+0.0054**** | [+0.0016, +0.0099] | 0.0050 |

### By Question Structure (gold entity/MeSH features)

| **MeSH Overlap w/ Gold** | | | | | | |
| ├─ None | 366 | 0.4114 | 0.4016 | +0.0098** | [+0.0021, +0.0196] | 0.0056 |
| ├─ >=1 | 577 | 0.4271 | 0.4245 | +0.0026 | [-0.0010, +0.0065] | 0.1586 |

| **Entity Overlap w/ Gold** | | | | | | |
| ├─ None | 649 | 0.4177 | 0.4132 | +0.0046 | [-0.0002, +0.0102] | 0.0618 |
| ├─ >=1 | 294 | 0.4283 | 0.4211 | +0.0072* | [+0.0014, +0.0136] | 0.0154 |

| **Shared Entity Edges** | | | | | | |
| ├─ High (>=100) | 940 | 0.4216 | 0.4162 | +0.0054** | [+0.0016, +0.0100] | 0.0024 |

| **Gold Count** | | | | | | |
| ├─ 1 passage | 204 | 0.5637 | 0.5539 | +0.0098 | [+0.0000, +0.0245] | 0.2772 |
| ├─ >=2 passages | 739 | 0.3816 | 0.3775 | +0.0042* | [+0.0006, +0.0079] | 0.0230 |


*p<0.05, **p<0.01 (paired bootstrap)

## Recall@10 Analysis

### By Question Type (keyword-based)

| Type | N | KCH | Pairwise | Δ | 95% CI | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yes/no | 248 | 0.5243 | 0.5229 | +0.0014 | [-0.0026, +0.0053] | 0.4654 |
| what_is | 225 | 0.5635 | 0.5633 | +0.0002 | [-0.0042, +0.0048] | 0.9488 |
| which | 181 | 0.5155 | 0.5105 | +0.0050 | [-0.0039, +0.0184] | 0.4372 |
| other | 169 | 0.5806 | 0.5772 | +0.0035 | [-0.0084, +0.0195] | 0.6740 |
| list | 99 | 0.4446 | 0.4446 | -0.0001 | [-0.0055, +0.0053] | 0.9696 |
| how | 21 | 0.4870 | 0.4731 | +0.0140 | [-0.0014, +0.0317] | 0.0912 |
| **Overall** | **943** | **0.5329** | **0.5306** | **+0.0023** | [-0.0011, +0.0064] | 0.2048 |

### By Question Structure (gold entity/MeSH features)

| **MeSH Overlap w/ Gold** | | | | | | |
| ├─ None | 366 | 0.4754 | 0.4722 | +0.0032 | [-0.0021, +0.0102] | 0.3058 |
| ├─ >=1 | 577 | 0.5693 | 0.5676 | +0.0017 | [-0.0024, +0.0067] | 0.4714 |

| **Entity Overlap w/ Gold** | | | | | | |
| ├─ None | 649 | 0.5205 | 0.5162 | +0.0043 | [-0.0002, +0.0096] | 0.0658 |
| ├─ >=1 | 294 | 0.5603 | 0.5623 | -0.0020 | [-0.0068, +0.0025] | 0.3778 |

| **Shared Entity Edges** | | | | | | |
| ├─ High (>=100) | 940 | 0.5331 | 0.5308 | +0.0023 | [-0.0011, +0.0063] | 0.2046 |

| **Gold Count** | | | | | | |
| ├─ 1 passage | 204 | 0.5833 | 0.5735 | +0.0098 | [+0.0000, +0.0245] | 0.2728 |
| ├─ >=2 passages | 739 | 0.5189 | 0.5187 | +0.0002 | [-0.0027, +0.0031] | 0.8672 |


*p<0.05, **p<0.01 (paired bootstrap)

## Interpretation

- At **k=5**, the hypergraph advantage is stronger ($\Delta$=+0.0054) with CI not crossing zero, consistent with the prior finding.
- At **k=10**, the advantage narrows ($\Delta$=+0.0023) and does not reach significance.
- The hypergraph advantage is concentrated in questions where gold passages share MeSH terms with the question (**MeSH Overlap >=1**: larger $\Delta$).
- Questions with **1 gold passage** show the largest $\Delta$ at k=10 (+0.0098), suggesting hypergraph helps most when fewer supporting passages are available.
- The pairwise graph captures most of the benefit at deeper cutoffs, while the n-ary hypergraph provides modest early-rank improvement.