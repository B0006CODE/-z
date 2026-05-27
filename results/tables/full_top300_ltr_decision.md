# Full Top-300 LTR Decision

Decision: do not run full top-300 LTR in this round.

## Evidence

| item | value |
| --- | --- |
| Previous PubTator baseline Recall@200 / Recall@300 | 0.7997 / 0.7997 |
| Previous PubTator baseline new gold @200 / @300 | 38 / 38 |
| Previous PubTator baseline noise @200 / @300 | 0.999064 / 0.999518 |
| Best filtered recovery setting | pubtator_low_df_0.01 |
| Best filtered recovery Recall@200 / Recall@300 | 0.7999 / 0.7999 |
| Best filtered recovery new gold @200 / @300 | 39 / 39 |
| Best filtered recovery noise @200 / @300 | 0.998941 / 0.999416 |
| Lowest-noise setting with new gold | pubtator_direct_only |
| Lowest-noise Recall@200 / Recall@300 | 0.7978 / 0.7978 |
| Lowest-noise new gold @200 / @300 | 27 / 27 |
| Lowest-noise noise @200 / @300 | 0.997940 / 0.998688 |
| Full hard subset size | 374 queries / 957 gold evidence |
| Hybrid top100 hard / expanded-only hard | 290 / 84 queries |

## Rationale

The filtered PubTator expansion only improves the previous PubTator baseline by one additional recovered gold evidence item on sample500, while noise remains above 0.999 at @300. The direct-only setting lowers noise below 0.999 at @300, but loses 12 recovered gold evidence items relative to the best filtered recovery setting.

The full hard subset diagnostic confirms that shared-cluster expansion can recover candidate-pool ceiling cases, but the signal is still broad and noisy. Expanded-only hard queries are heavily concentrated in sparse lexical/entity conditions, especially entity-overlap-zero cases, and the available PubTator sample contributes only a very small number of new gold reasons in the full hard diagnostic.

Given the previous sample500 LTR result where Full KCH did not beat Remove-hypergraph, the current expansion evidence is not strong enough to justify a full top-300 LTR run.
