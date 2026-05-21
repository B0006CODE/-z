# Reranking Failure Analysis

Comparison: Enhanced Hybrid w122 -> Enhanced KCH-MedRank; top-k = 10.

| diagnostic | value |
| --- | ---: |
| paired_questions | 943 |
| questions_with_lost_gold | 214 |
| lost_gold_passages | 307 |
| questions_with_rescued_gold | 414 |
| rescued_gold_passages | 661 |

## Lost Gold Evidence

| question_id | passage_id | baseline rank | candidate rank | rank delta | MeSH overlap |
| --- | --- | ---: | ---: | ---: | --- |
| 1599 | 16154111 | 2 | 18 | 16 | none |
| 124 | 15148346 | 2 | 12 | 10 | thyroid hormones |
| 154 | 2807143 | 3 | 18 | 15 | thyroid hormones |
| 414 | 23728220 | 3 | 18 | 15 | none |
| 3609 | 29515069 | 3 | 14 | 11 | dasatinib |
| 934 | 16821517 | 3 | 12 | 9 | gastrointestinal stromal tumors |
| 124 | 17710084 | 3 | 11 | 8 | thyroid hormones |
| 1274 | 21438918 | 3 | 11 | 8 | fanconi anemia |

### Lost Gold Evidence 1: 1599

Question: Is there any protein that undergoes both mono-ubiquitination and poly-ubiquitination?

Gold passage: We previously reported that oxidative stress is associated with unloading-mediated ubiquitination of muscle proteins. To further elucidate the involvement of oxidative stress in ubiquitination, we examined the ubiquitination profile in rat myoblastic L6 cells after treatment with hydrogen peroxide. Hydrogen peroxide induced many ubiquitinated proteins wit...

### Lost Gold Evidence 2: 124

Question: Is DITPA a thyroid hormone analog utilized in experimental and clinical studies

Gold passage: The heart is an important target of thyroid hormone actions. Only a limited number of cardiac target genes have been identified, and little is known about their regulation by T(3) (3,3',5-triiodothyronine) and thyroid hormone analogs. We used an oligonucleotide microarray to identify novel cardiac genes regulated by T(3) and two thyroid hormone analogs, 3...

### Lost Gold Evidence 3: 154

Question: Does strenuous physical activity affect thyroid hormone metabolism?

Gold passage: Significant increases in the concentration of plasma glucagon-like immunoreactivity (GLI) and plasma levels of free fatty acids (FFA) and triglycerides (TG) concomitant with decreases in circulating levels of thyroxine (T4) and triiodothyronine (T3) and T3/T4 ratio were observed in homing pigeons, untrained for 3 months, after a flight of 48 km lasting 90...

### Lost Gold Evidence 4: 414

Question: List receptors of the drug Cilengitide

Gold passage: The prognosis of children with high-grade glioma or high-risk neuroblastoma remains poor. Cilengitide is a selective antagonist of αvβ3 and αvβ5 integrins, which are involved in tumor growth and development of metastasis. We have evaluated the effects of cilengitide on pediatric glioma and neuroblastoma cell lines for the first time. Expression levels of...

### Lost Gold Evidence 5: 3609

Question: Which disease is Dasatinib used to treat?

Gold passage: A 37-year-old woman was diagnosed with chronic phase chronic myeloid leukemia. Nilotinib treatment was initiated; however, it had to be discontinued due to an allergic reaction one month later, and dasatinib treatment was provided. Although favorable response was obtained, she started complaining of shortness of breath 7 months after initiating dasatinib...

### Lost Gold Evidence 6: 934

Question: What are the treatments of choice for GIST (gastrointestinal stromal tumor)?

Gold passage: Malignant gastointestinal stromal tumors (M-GIST) are rare mesenchymal tumors that arise in the wall of the gastrointestinal (GI) tract. Small intestinal GIST account for approximately 35% of all GIST the diagnosis of these tumors is difficult to establish, because the symptoms are vague and non-specific and traditional endoscopy is commonly unsatisfactor...

### Lost Gold Evidence 7: 124

Question: Is DITPA a thyroid hormone analog utilized in experimental and clinical studies

Gold passage: Thyroid hormone (T3 and T4) has many beneficial effects including enhancing cardiac function, promoting weight loss and reducing serum cholesterol. Excess thyroid hormone is, however, associated with unwanted effects on the heart, bone and skeletal muscle. We therefore need analogs that harness the beneficial effects of thyroid hormone without the untowar...

### Lost Gold Evidence 8: 1274

Question: Is Fanconi anemia presented as a genetically and clinically heterogeneous disease entity?

Gold passage: Bednar tumor is a rare pigmented variation of dermatofibrosarcoma protuberans, present in 1 to 5% of all patients with dermatofibrosarcoma protuberans. No significant clinicopathologic differences exist between Bednar tumor and conventional dermatofibrosarcoma protuberans apart from the presence of scattered nonneoplastic pigmented dendritic cells in the...

## Rescued Gold Evidence

| question_id | passage_id | baseline rank | candidate rank | rank delta | MeSH overlap |
| --- | --- | ---: | ---: | ---: | --- |
| 4614 | 33594055 | 54 | 1 | -53 | bacteriophages |
| 3924 | 26083752 | 52 | 1 | -51 | none |
| 3049 | 29358629 | 46 | 1 | -45 | ploidies, telomere, whole genome sequencing |
| 3489 | 15850899 | 43 | 1 | -42 | meningioma, radiosurgery |
| 3959 | 31187503 | 34 | 1 | -33 | mutation, polyneuropathies |
| 4194 | 31953314 | 19 | 1 | -18 | adenosine, neoplasms |
| 3004 | 23625205 | 18 | 1 | -17 | none |
| 3064 | 25852060 | 17 | 1 | -16 | none |

### Rescued Gold Evidence 1: 4614

Question: What is the p-crAssphage?

Gold passage: CrAssphage is the most abundant human-associated virus and the founding member of a large group of bacteriophages, discovered in animal-associated and environmental metagenomes, that infect bacteria of the phylum Bacteroidetes. We analyze 4907 Circular Metagenome Assembled Genomes (cMAGs) of putative viruses from human gut microbiomes and identify nearly...

### Rescued Gold Evidence 2: 3924

Question: What is the function of ketohexokinase-A?

Gold passage: Fructose is a major component of dietary sugar and its overconsumption exacerbates key pathological features of metabolic syndrome. The central fructose-metabolising enzyme is ketohexokinase (KHK), which exists in two isoforms: KHK-A and KHK-C, generated through mutually exclusive alternative splicing of KHK pre-mRNAs. KHK-C displays superior affinity for...

### Rescued Gold Evidence 3: 3049

Question: Which ploidy-agnostic method has been developed for estimating telomere length from whole genome sequencing data?

Gold passage: nan

### Rescued Gold Evidence 4: 3489

Question: Can radiation induced meningiomas be treated with radiosurgery?

Gold passage: nan

### Rescued Gold Evidence 5: 3959

Question: Are PDXK mutations linked to polyneuropathy?

Gold passage: nan

### Rescued Gold Evidence 6: 4194

Question: Is adenosine signaling prognostic for cancer outcome?

Gold passage: nan

### Rescued Gold Evidence 7: 3004

Question: What is the role of metalloproteinase-17 (ADAM17) in NK cells?

Gold passage: A disintegrin and metalloproteinase-17 (ADAM17) is a member of the metalloproteinase superfamily and involved in the cleavage of ectodomain of many transmembrane proteins. ADAM17 is overexpressed in a variety of human tumors, which is associated with tumor development and progression. In the present study, we sought to investigate the expression and funct...

### Rescued Gold Evidence 8: 3064

Question: What is the mechanism of the drug CRT0066101?

Gold passage: Invasive ductal carcinomas (IDC) of the breast are associated with altered expression of hormone receptors (HR), amplification or overexpression of HER2, or a triple-negative phenotype. The most aggressive cases of IDC are characterized by a high proliferation rate, a great propensity to metastasize, and their ability to resist to standard chemotherapy, h...
