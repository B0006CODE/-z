# Interpretability Analysis: Three-Mechanism Classification

## Summary
- Rescued (outside→inside top-10): 648 gold passages
- Lost (inside→outside top-10): 306 gold passages

### Three Interpretable Mechanisms
1. **MeSH Hierarchy Path**: Question and passage share MeSH concepts through hierarchical tree paths
2. **Shared Entity Cluster Path**: Gold passage and other candidates share biomedical entities, forming hyperedge clusters
3. **PrimeKG Relation Path**: Question entities are relationally linked to passage entities via PrimeKG
4. **Diffusion Only**: No explicit knowledge alignment; rescue from hypergraph topology and diffusion propagation alone

## Rescued Cases: Mechanism Distribution
Total rescued cases analyzed: 648

| Mechanism | Count | % | Avg Rank Gain | Q-Type Breakdown |
| --- | ---: | ---: | ---: | --- |
| mesh_hierarchy | 492 | 75.9 | 22.1 | yes/no: 407, which: 32, list/synthesis: 30, other: 17 |
| entity_cluster | 155 | 23.9 | 14.0 | yes/no: 131, list/synthesis: 9, which: 7, other: 3 |
| relation_path | 1 | 0.2 | 25.0 | yes/no: 1 |
| diffusion_only | 0 | 0.0 | 0.0 |  |

## Lost Cases: Mechanism Distribution
Total lost cases analyzed: 306

| Mechanism | Count | % | Avg Rank Loss | Q-Type Breakdown |
| --- | ---: | ---: | ---: | --- |
| mesh_hierarchy | 268 | 87.6 | 6.8 | yes/no: 228, list/synthesis: 26, which: 7, treatment: 5 |
| entity_cluster | 38 | 12.4 | 5.6 | yes/no: 33, list/synthesis: 2, other: 2, mechanism: 1 |
| relation_path | 0 | 0.0 | 0.0 |  |
| diffusion_only | 0 | 0.0 | 0.0 |  |

## Mechanism Activation Statistics (Rescued Cases)

| Metric | MeSH Hierarchy | Entity Cluster | Relation Path | Diffusion Only |
| --- | ---: | ---: | ---: | ---: |
| mesh_overlap_count | 1.18 | 0.03 | 0.00 | 0.00 |
| entity_overlap_count | 0.09 | 0.48 | 0.00 | 0.00 |
| question_entity_coverage | 0.06 | 0.36 | 0.00 | 0.00 |
| relation_count | 0.01 | 0.01 | 1.00 | 0.00 |
| shared_entity_edges | 126.59 | 127.84 | 128.00 | 0.00 |
| shared_mesh_parent_size | 48.85 | 15.05 | 0.00 | 0.00 |

## Path Tracing: Top Rescued Cases by Mechanism

### Top mesh_hierarchy Cases (largest rank gains)

| Q-ID | P-ID | Rank Gain | Q-Type | Question |
| --- | --- | ---: | --- | --- |
| 3204 | 28801534 | 83 | yes/no | What is the function of the cGAS pathway? |
| 1669 | 23576609 | 65 | yes/no | Does ghrelin play a role in ischemic stroke? |
| 1669 | 24768795 | 64 | yes/no | Does ghrelin play a role in ischemic stroke? |
| 329 | 17929114 | 63 | yes/no | What tyrosine kinase, involved in a Philadelphia- chromosome positive chronic myelogenous leukemia,  |
| 1084 | 17640894 | 62 | yes/no | Which is the methyl donor of histone methyltransferases? |

### Top entity_cluster Cases (largest rank gains)

| Q-ID | P-ID | Rank Gain | Q-Type | Question |
| --- | --- | ---: | --- | --- |
| 4589 | 34798793 | 84 | which | A combination of which two drugs was tested in the IMbrave150 trial? |
| 3999 | 28240610 | 64 | yes/no | What is another name for the drug AMG334? |
| 3064 | 26797128 | 56 | yes/no | What is the mechanism of the drug CRT0066101? |
| 4614 | 33715349 | 56 | yes/no | What is the p-crAssphage? |
| 779 | 19016324 | 53 | yes/no | Which are the cardiac effects of thyronamines? |

### Top relation_path Cases (largest rank gains)

| Q-ID | P-ID | Rank Gain | Q-Type | Question |
| --- | --- | ---: | --- | --- |
| 3269 | 30588330 | 25 | yes/no | Is Li–Fraumeni syndrome a rare, autosomal recessive, hereditary disorder that predisposes carriers t |

## Path Tracing: Top Lost Cases by Mechanism

### Top mesh_hierarchy Lost Cases (largest rank losses)

| Q-ID | P-ID | Rank Loss | Q-Type | Question |
| --- | --- | ---: | --- | --- |
| 149 | 8725589 | 46 | yes/no | Which are the drugs utilized for the burning mouth syndrome? |
| 849 | 26089446 | 41 | yes/no | Which calcium channels does ethosuximide target? |
| 149 | 9844361 | 40 | yes/no | Which are the drugs utilized for the burning mouth syndrome? |
| 2819 | 24333266 | 38 | yes/no | Which metabolic pathways have been associated with Systemic Lupus Erythematosus? |
| 3569 | 29542093 | 29 | yes/no | PDQ39 questionnaires is design for which disease? |

### Top entity_cluster Lost Cases (largest rank losses)

| Q-ID | P-ID | Rank Loss | Q-Type | Question |
| --- | --- | ---: | --- | --- |
| 2944 | 30201700 | 18 | list/synthesis | List phagosomal markers. |
| 3519 | 30079523 | 16 | yes/no | What molecules are the multidrug transporter MDR3 targeting? |
| 4104 | 31357172 | 13 | yes/no | Which are the main advantages of kallisto against similar methodologies? |
| 3569 | 28770096 | 12 | yes/no | PDQ39 questionnaires is design for which disease? |
| 1524 | 19296976 | 11 | yes/no | What is the gold standard treatment for Iatrogenic male incontinence? |


## Mechanism Co-occurrence Analysis (Rescued Cases)

Cases where multiple mechanisms simultaneously contributed to rescue:

| Active Mechanisms | Count | % |
| --- | ---: | ---: |
| mesh_hierarchy + entity_cluster | 322 | 49.7 |
| mesh_hierarchy + entity_cluster + relation_path | 256 | 39.5 |
| entity_cluster + relation_path | 38 | 5.9 |
| entity_cluster | 32 | 4.9 |