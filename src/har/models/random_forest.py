"""Random Forest baseline on the 561 engineered features."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from har.config import RFConfig


def build_random_forest(cfg: RFConfig, seed: int, **overrides) -> RandomForestClassifier:
    params = {
        "n_estimators": cfg.n_estimators,
        "max_depth": cfg.max_depth,
        "min_samples_leaf": cfg.min_samples_leaf,
        "max_features": cfg.max_features,
        "n_jobs": cfg.n_jobs,
        **overrides,
    }
    return RandomForestClassifier(random_state=seed, **params)


def count_nodes(model: RandomForestClassifier) -> int:
    """Total decision nodes across all trees - the RF analogue of a parameter count."""
    return int(sum(est.tree_.node_count for est in model.estimators_))
