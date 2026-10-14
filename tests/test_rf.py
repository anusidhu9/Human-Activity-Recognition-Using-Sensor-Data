"""Stage 4 tests: metrics, error categories and an end-to-end RF run on synthetic data."""

import json

import numpy as np
import pytest
from sqlalchemy import text

from har.config import RFConfig
from har.data.pipeline import prepare_data
from har.db import repository as repo
from har.db.ingest import ingest_metadata
from har.db.schema import create_schema, get_engine
from har.evaluation.error_analysis import error_category
from har.evaluation.metrics import compute_metrics, metrics_to_rows
from har.training.sklearn_runner import run_random_forest

MANIFEST = {"source_url": "x", "sha256": "1" * 64, "size_bytes": 1,
            "downloaded_at": "2026-10-14T00:00:00+00:00"}


def test_compute_metrics_known_values():
    y_true = np.array([1, 1, 2, 2, 4, 5])
    y_pred = np.array([1, 2, 2, 2, 5, 5])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"][1]["recall"] == pytest.approx(0.5)
    assert m["per_class"][2]["precision"] == pytest.approx(2 / 3)
    assert m["confusion_matrix"].shape == (6, 6)
    assert m["confusion_matrix"].sum() == 6
    assert m["confusion_matrix"][3, 4] == 1  # sitting (4) predicted as standing (5)


def test_metric_rows_cover_overall_and_per_class():
    m = compute_metrics(np.arange(1, 7), np.arange(1, 7))
    rows = metrics_to_rows(m)
    assert len(rows) == 1 + 2 * 3 + 6 * 3
    assert all(r["value"] == pytest.approx(1.0) for r in rows)


@pytest.mark.parametrize("t,p,cat", [(4, 5, "static_static"), (1, 3, "dynamic_dynamic"),
                                     (2, 6, "static_dynamic"), (3, 3, None)])
def test_error_category(t, p, cat):
    assert error_category(t, p) == cat


def test_rf_end_to_end_on_synthetic_data(tmp_path, synthetic_splits, synthetic_dataset_dir):
    # make the features informative so the model has something to learn
    rng = np.random.default_rng(0)
    for s in synthetic_splits.values():
        s.features[:] = s.labels[:, None] / 6 + rng.normal(0, 0.01, s.features.shape)
    engine = get_engine(tmp_path / "rf.sqlite")
    create_schema(engine)
    did = ingest_metadata(engine, synthetic_splits, MANIFEST, dataset_dir=synthetic_dataset_dir)
    data = prepare_data("features", n_val_subjects=2,
                        train=synthetic_splits["train"], test=synthetic_splits["test"])
    cfg = RFConfig(n_estimators=10, n_jobs=1, grid={"min_samples_leaf": [1, 2]})

    result = run_random_forest(data, engine, did, seed=0, cfg=cfg,
                               checkpoints_dir=tmp_path / "ckpt")

    assert result["test"]["accuracy"] == pytest.approx(1.0)
    ckpt = result["checkpoint"]
    assert (ckpt / "model.joblib").exists()
    assert len(json.loads((ckpt / "config.json").read_text())["hyperparams"]["grid_results"]) == 2
    with engine.connect() as c:
        status, train_t, infer_t = c.execute(text(
            "SELECT status, train_time_s, inference_time_s FROM experiments")).one()
        n_test_preds = c.execute(text(
            "SELECT COUNT(*) FROM predictions WHERE eval_split = 'test'")).scalar()
    assert status == "completed" and train_t > 0 and infer_t > 0
    assert n_test_preds == len(synthetic_splits["test"])


def test_rf_rejects_window_input(tmp_path, synthetic_splits):
    data = prepare_data("windows", n_val_subjects=2,
                        train=synthetic_splits["train"], test=synthetic_splits["test"])
    with pytest.raises(ValueError):
        run_random_forest(data, engine=None, dataset_id=1, seed=0,
                          checkpoints_dir=tmp_path)
