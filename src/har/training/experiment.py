"""Model-agnostic pieces of a run: inference timing and logging results to SQL.

Both the scikit-learn and the PyTorch runners use these, so every model is
timed and evaluated in exactly the same way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from sqlalchemy.engine import Engine

from har.db import repository as repo
from har.evaluation.error_analysis import annotate_errors
from har.evaluation.metrics import LABEL_IDS, compute_metrics, metrics_to_rows
from har.utils import Timer


def time_inference(predict_proba: Callable[[np.ndarray], np.ndarray], X: np.ndarray,
                   warmup: int = 64) -> tuple[np.ndarray, float]:
    """Run ``predict_proba`` on all of ``X`` after a small warm-up call.

    Returns (probs, seconds). The warm-up excludes one-off costs (thread pool
    start-up, lazy initialisation) from the measurement.
    """
    predict_proba(X[:warmup])
    with Timer() as t:
        probs = predict_proba(X)
    return probs, t.elapsed


def probs_to_labels(probs: np.ndarray) -> np.ndarray:
    """(N, 6) probabilities in activity-id order -> predicted activity ids 1..6."""
    return np.asarray(LABEL_IDS)[probs.argmax(axis=1)]


def log_evaluation(engine: Engine, experiment_id: int, eval_split: str,
                   sample_ids: np.ndarray, y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Compute metrics and store predictions, probabilities, metrics and error categories."""
    y_pred = probs_to_labels(probs)
    m = compute_metrics(y_true, y_pred)
    repo.save_predictions(engine, experiment_id, eval_split, sample_ids, probs)
    repo.save_metrics(engine, experiment_id, eval_split, metrics_to_rows(m))
    if eval_split == "test":  # error analysis is about the held-out test subjects
        repo.save_error_annotations(engine, experiment_id,
                                    annotate_errors(sample_ids, y_true, y_pred))
    return m


def metrics_json(m: dict) -> dict:
    """JSON-serialisable copy of ``compute_metrics`` output (for the checkpoint folder)."""
    return {
        "accuracy": float(m["accuracy"]),
        "macro": {k: float(v) for k, v in m["macro"].items()},
        "weighted": {k: float(v) for k, v in m["weighted"].items()},
        "per_class": {str(a): {k: float(v) for k, v in s.items()}
                      for a, s in m["per_class"].items()},
        "confusion_matrix": m["confusion_matrix"].tolist(),
    }


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))
