"""Categorize misclassifications by the kind of confusion.

Activities fall into two groups: dynamic (the three walking types) and static
(sitting, standing, laying). Errors within a group (e.g. sitting vs standing)
are expected to be far more common than errors across groups.
"""

from __future__ import annotations

import numpy as np

DYNAMIC = {1, 2, 3}  # walking, upstairs, downstairs
STATIC = {4, 5, 6}  # sitting, standing, laying


def error_category(true_id: int, pred_id: int) -> str | None:
    """Return the confusion category, or None if the prediction is correct."""
    if true_id == pred_id:
        return None
    if true_id in STATIC and pred_id in STATIC:
        return "static_static"
    if true_id in DYNAMIC and pred_id in DYNAMIC:
        return "dynamic_dynamic"
    return "static_dynamic"


def annotate_errors(sample_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """One annotation row per misclassified sample."""
    rows = []
    for sid, t, p in zip(sample_ids, y_true, y_pred):
        cat = error_category(int(t), int(p))
        if cat is not None:
            rows.append({"sample_id": int(sid), "category": cat})
    return rows
