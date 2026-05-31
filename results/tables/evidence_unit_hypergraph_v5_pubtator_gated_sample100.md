| method | split | recall@10 | mrr@10 | ndcg@10 | recall@100 | delta_recall@10 | delta_mrr@10 | delta_ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Source candidate order | all_selected | 0.4897 | 0.8817 | 0.6630 | 0.7972 |  |  |  |
| Source candidate order | held_out_test | 0.4467 | 0.7817 | 0.5940 | 0.7655 |  |  |  |
| retrieval_ltr | all_selected | 0.4892 | 0.8812 | 0.6672 | 0.7972 | -0.0005 | -0.0005 | +0.0041 |
| retrieval_ltr | held_out_test | 0.4235 | 0.7767 | 0.5775 | 0.7655 | -0.0232 | -0.0050 | -0.0165 |
| flat_evidence_unit_ltr | all_selected | 0.4987 | 0.8812 | 0.6693 | 0.7972 | +0.0091 | -0.0005 | +0.0063 |
| flat_evidence_unit_ltr | held_out_test | 0.4323 | 0.7725 | 0.5803 | 0.7655 | -0.0145 | -0.0092 | -0.0137 |
| evidence_unit_hypergraph_ltr | all_selected | 0.4945 | 0.8812 | 0.6649 | 0.7972 | +0.0048 | -0.0005 | +0.0019 |
| evidence_unit_hypergraph_ltr | held_out_test | 0.4298 | 0.7767 | 0.5800 | 0.7655 | -0.0170 | -0.0050 | -0.0141 |
| without_evidence_quality | all_selected | 0.4948 | 0.8812 | 0.6655 | 0.7972 | +0.0051 | -0.0005 | +0.0024 |
| without_evidence_quality | held_out_test | 0.4298 | 0.7767 | 0.5796 | 0.7655 | -0.0170 | -0.0050 | -0.0144 |
| without_major_mesh | all_selected | 0.4945 | 0.8812 | 0.6649 | 0.7972 | +0.0048 | -0.0005 | +0.0019 |
| without_major_mesh | held_out_test | 0.4298 | 0.7767 | 0.5800 | 0.7655 | -0.0170 | -0.0050 | -0.0141 |
| without_cluster_support | all_selected | 0.4991 | 0.8820 | 0.6706 | 0.7972 | +0.0094 | +0.0003 | +0.0076 |
| without_cluster_support | held_out_test | 0.4373 | 0.7767 | 0.5844 | 0.7655 | -0.0095 | -0.0050 | -0.0096 |
| gated_evidence_unit_hypergraph | all_selected | 0.4897 | 0.8817 | 0.6631 | 0.7972 | +0.0000 | +0.0000 | +0.0001 |
| gated_evidence_unit_hypergraph | held_out_test | 0.4467 | 0.7817 | 0.5940 | 0.7655 | +0.0000 | +0.0000 | +0.0000 |
| hyperpath_score_only | all_selected | 0.3220 | 0.5260 | 0.3924 | 0.7972 | -0.1677 | -0.3557 | -0.2707 |
| hyperpath_score_only | held_out_test | 0.2830 | 0.4627 | 0.3726 | 0.7655 | -0.1637 | -0.3190 | -0.2214 |
