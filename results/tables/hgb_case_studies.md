# HGB Reranking Case Studies

Selection rule: prefer gold evidence moved from outside top-10 by Hybrid RRF into top-10 by HGB; otherwise use the largest positive gold-rank gain.

## BioASQ

Questions analyzed: 943; rescued to top-10: 187 questions / 241 gold passages; lost from top-10: 119 questions / 147 gold passages; improved min-gold-rank: 70; worsened: 36; no gold in HGB top100: 121.

| question_id | passage_id | Hybrid gold rank | HGB gold rank | rank gain | entity overlap | MeSH overlap | key HGB features |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 694 | 22162803 | 32 | 5 | 27 | hiv | none | hypergraph=0.118, entity_cov=0.500, mesh_cov=0.000, shared_edges=128, mesh_edges=93 |
| 1724 | 19214283 | 35 | 10 | 25 | none | colon, colonoscopy | hypergraph=0.081, entity_cov=0.000, mesh_cov=1.000, shared_edges=128, mesh_edges=90 |
| 1479 | 10790371 | 29 | 5 | 24 | none | none | hypergraph=0.200, entity_cov=0.000, mesh_cov=0.000, shared_edges=128, mesh_edges=97 |

### BioASQ case 1: 694

Question: Is there a phylogenetic analysis for HIV?

Gold passage: The South American human immunodeficiency virus type 1 (HIV-1) epidemic is driven by several subtypes (B, C, and F1) and circulating and unique recombinant forms derived from those subtypes. Those variants are heterogeneously distributed around the continent in a country-specific manner. Despite some inconsistencies mainly derived from sampling biases and...

### BioASQ case 2: 1724

Question: What colonoscopy findings have been reported in autism

Gold passage: Les troubles du spectre autistique font référence à des syndromes de gravité diverse, caractérisés par une perturbation des interactions sociales, des retards sur le plan de la communication, ainsi que des comportements et des intérêts limités et répétés. La prévalence des troubles du spectre autistique est en hausse, tandis que leur étiologie reste impré...

### BioASQ case 3: 1479

Question: what is the role of MEF-2 in cardiomyocyte differentiation?

Gold passage: The myocyte enhancer factor-2 (MEF2) proteins are MADS-box transcription factors that are essential for differentiation of all muscle lineages but their mechanisms of action remain largely undefined. In mammals, the earliest site of MEF2 expression is the heart where the MEF2C isoform is detectable as early as embryonic day 7.5. Inactivation of the MEF2C...

## PubMedQA

Questions analyzed: 200; rescued to top-10: 48 questions / 58 gold passages; lost from top-10: 7 questions / 7 gold passages; improved min-gold-rank: 2; worsened: 2; no gold in HGB top100: 0.

| question_id | passage_id | Hybrid gold rank | HGB gold rank | rank gain | entity overlap | MeSH overlap | key HGB features |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| pubmedqa_20382292 | pubmedqa_20382292_3 | 100 | 7 | 93 | none | knee | hypergraph=0.010, entity_cov=0.000, mesh_cov=1.000, shared_edges=38, mesh_edges=100 |
| pubmedqa_26606599 | pubmedqa_26606599_3 | 79 | 5 | 74 | none | acetabulum, ossification heterotopic | hypergraph=0.435, entity_cov=0.000, mesh_cov=1.000, shared_edges=30, mesh_edges=100 |
| pubmedqa_16216859 | pubmedqa_16216859_2 | 80 | 8 | 72 | none | collateral circulation, coronary restenosis | hypergraph=0.129, entity_cov=0.000, mesh_cov=0.400, shared_edges=40, mesh_edges=100 |

### PubMedQA case 1: pubmedqa_20382292

Question: Knee extensor strength, dynamic stability, and functional ambulation: are they related in Parkinson's disease?

Gold passage: PMID 20382292 PARTICIPANTS Patients (N=44) with idiopathic PD.

### PubMedQA case 2: pubmedqa_26606599

Question: Do Surrogates of Injury Severity Influence the Occurrence of Heterotopic Ossification in Fractures of the Acetabulum?

Gold passage: PMID 26606599 PARTICIPANTS Two hundred forty-one patients who were treated through a posterior approach with a minimum of 6-month radiographic follow-up were identified from an acetabular fracture database.

### PubMedQA case 3: pubmedqa_16216859

Question: Does a well developed collateral circulation predispose to restenosis after percutaneous coronary intervention?

Gold passage: PMID 16216859 PATIENTS AND SETTING 58 patients undergoing elective single vessel PCI in a tertiary referral interventional cardiac unit in the UK.
