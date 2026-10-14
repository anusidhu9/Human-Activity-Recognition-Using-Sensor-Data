"""Read/write helpers used by the training and evaluation code.

Experiment results are only ever appended: a run is created once (unique
``run_id``) and its predictions/metrics are inserted, never replaced.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
from sqlalchemy import func, insert, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from har.config import ACTIVITY_LABELS
from har.db.schema import Dataset, Experiment, Metric, Prediction, PredictionProb, Sample


def latest_dataset_id(engine: Engine) -> int:
    with Session(engine) as s:
        dataset_id = s.scalar(select(func.max(Dataset.dataset_id)))
    if dataset_id is None:
        raise LookupError("No dataset ingested yet - run scripts/build_db.py first.")
    return dataset_id


def sample_ids(engine: Engine, dataset_id: int, split: str) -> np.ndarray:
    """``sample_id`` for every row of an official split, ordered by ``row_index``.

    Index this with the same row indices used to slice the arrays, so each
    prediction is stored against the right sample.
    """
    with Session(engine) as s:
        ids = s.scalars(select(Sample.sample_id)
                        .where(Sample.dataset_id == dataset_id, Sample.split == split)
                        .order_by(Sample.row_index)).all()
    return np.asarray(ids, dtype=np.int64)


def create_experiment(engine: Engine, *, run_id: str, dataset_id: int, model_type: str,
                      input_representation: str, hyperparams: dict, seed: int, device: str,
                      val_subjects: list[int], notes: str | None = None) -> int:
    with Session(engine) as s, s.begin():
        exp = Experiment(run_id=run_id, dataset_id=dataset_id, model_type=model_type,
                         input_representation=input_representation,
                         hyperparams_json=json.dumps(hyperparams, sort_keys=True),
                         seed=seed, device=device,
                         val_subjects_json=json.dumps(sorted(int(v) for v in val_subjects)),
                         notes=notes)
        s.add(exp)
        s.flush()
        return exp.experiment_id


def finish_experiment(engine: Engine, experiment_id: int, *, status: str = "completed",
                      **fields) -> None:
    """Set final fields (timings, n_params, checkpoint_path...) and the status."""
    with Session(engine) as s, s.begin():
        exp = s.get(Experiment, experiment_id)
        for key, value in fields.items():
            if not hasattr(Experiment, key):
                raise AttributeError(f"Experiment has no column {key!r}")
            setattr(exp, key, value)
        exp.status = status
        exp.finished_at = datetime.now()


def save_predictions(engine: Engine, experiment_id: int, eval_split: str,
                     sample_ids_: np.ndarray, probs: np.ndarray) -> None:
    """Store predicted class, its confidence and the full probability vector.

    ``probs`` is (N, 6) with columns ordered by activity id 1..6.
    """
    if probs.shape != (len(sample_ids_), len(ACTIVITY_LABELS)):
        raise ValueError(f"probs shape {probs.shape} does not match {len(sample_ids_)} samples")
    activity_ids = np.array(list(ACTIVITY_LABELS))
    pred_idx = probs.argmax(axis=1)
    with Session(engine) as s, s.begin():
        s.execute(insert(Prediction), [
            {"experiment_id": experiment_id, "sample_id": int(sid), "eval_split": eval_split,
             "pred_activity_id": int(activity_ids[k]), "confidence": float(probs[i, k])}
            for i, (sid, k) in enumerate(zip(sample_ids_, pred_idx))
        ])
        s.execute(insert(PredictionProb), [
            {"experiment_id": experiment_id, "sample_id": int(sid),
             "activity_id": int(aid), "prob": float(probs[i, j])}
            for i, sid in enumerate(sample_ids_) for j, aid in enumerate(activity_ids)
        ])


def save_metrics(engine: Engine, experiment_id: int, eval_split: str, rows: list[dict]) -> None:
    """Insert metric rows: dicts with metric_name, value, and optional average/activity_id."""
    with Session(engine) as s, s.begin():
        s.execute(insert(Metric), [
            {"experiment_id": experiment_id, "eval_split": eval_split,
             "average": r.get("average"), "activity_id": r.get("activity_id"),
             "metric_name": r["metric_name"], "value": float(r["value"])}
            for r in rows
        ])
