from __future__ import annotations

from lightgbm import LGBMRanker


def make_lambdamart_ranker(
    *,
    seed: int,
    num_leaves: int = 15,
    learning_rate: float = 0.05,
    n_estimators: int = 80,
) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
