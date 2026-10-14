"""Stage 3 tests: schema, metadata ingestion, repository helpers and views."""

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from har.db import repository as repo
from har.db.ingest import ingest_metadata
from har.db.schema import create_schema, get_engine

MANIFEST = {"source_url": "https://example.org/har.zip", "sha256": "0" * 64,
            "size_bytes": 123, "downloaded_at": "2026-10-14T00:00:00+00:00"}


@pytest.fixture
def engine(tmp_path, synthetic_splits, synthetic_dataset_dir):
    eng = get_engine(tmp_path / "test.sqlite")
    create_schema(eng)
    ingest_metadata(eng, synthetic_splits, MANIFEST, dataset_dir=synthetic_dataset_dir)
    return eng


def _count(engine, sql):
    with engine.connect() as c:
        return c.execute(text(sql)).scalar()


def test_metadata_counts(engine, synthetic_splits):
    n_train, n_test = (len(synthetic_splits[s]) for s in ("train", "test"))
    assert _count(engine, "SELECT COUNT(*) FROM activities") == 6
    assert _count(engine, "SELECT COUNT(*) FROM subjects") == 8
    assert _count(engine, "SELECT COUNT(*) FROM samples") == n_train + n_test
    # features + labels + subjects + 9 channels, per split
    assert _count(engine, "SELECT COUNT(*) FROM data_files") == 2 * 12


def test_samples_store_no_sensor_values(engine):
    with engine.connect() as c:
        cols = {r[1] for r in c.execute(text("PRAGMA table_info(samples)"))}
    assert cols == {"sample_id", "dataset_id", "split", "row_index", "subject_id", "activity_id"}


def test_sample_ids_follow_row_order(engine, synthetic_splits):
    did = repo.latest_dataset_id(engine)
    ids = repo.sample_ids(engine, did, "test")
    assert len(ids) == len(synthetic_splits["test"])
    with engine.connect() as c:
        rows = [r[0] for r in c.execute(
            text("SELECT row_index FROM samples WHERE sample_id IN (%s)" % ",".join(map(str, ids))
                 + " ORDER BY sample_id"))]
    assert rows == list(range(len(ids)))


def test_foreign_keys_enforced(engine):
    with pytest.raises(IntegrityError), engine.begin() as c:
        c.execute(text("INSERT INTO samples (dataset_id, split, row_index, subject_id, activity_id)"
                       " VALUES (1, 'test', 9999, 1, 99)"))


def _fake_experiment(engine, run_id="rf_s42_x"):
    return repo.create_experiment(
        engine, run_id=run_id, dataset_id=repo.latest_dataset_id(engine), model_type="rf",
        input_representation="features_561", hyperparams={"n_estimators": 10}, seed=42,
        device="cpu", val_subjects=[1, 3])


def test_run_id_is_unique(engine):
    _fake_experiment(engine)
    with pytest.raises(IntegrityError):
        _fake_experiment(engine)


def test_predictions_and_views(engine, synthetic_splits):
    exp = _fake_experiment(engine)
    did = repo.latest_dataset_id(engine)
    ids = repo.sample_ids(engine, did, "test")
    labels = synthetic_splits["test"].labels
    # predict the true class everywhere except the first two samples
    probs = np.full((len(ids), 6), 0.02)
    probs[np.arange(len(ids)), labels - 1] = 0.9
    probs[0], probs[1] = np.roll(probs[0], 1), np.roll(probs[1], 1)
    repo.save_predictions(engine, exp, "test", ids, probs)
    repo.save_metrics(engine, exp, "test", [{"metric_name": "accuracy", "value": 0.9}])
    repo.finish_experiment(engine, exp, train_time_s=1.5, n_params=0)

    assert _count(engine, "SELECT COUNT(*) FROM v_misclassified") == 2
    assert _count(engine, "SELECT COUNT(*) FROM prediction_probs") == len(ids) * 6
    assert _count(engine, "SELECT SUM(n) FROM v_confusion_matrix") == len(ids)
    assert _count(engine, "SELECT COUNT(*) FROM v_per_subject_accuracy") == 2
    assert _count(engine, "SELECT status FROM experiments") == "completed"
    assert _count(engine, "SELECT value FROM metrics") == pytest.approx(0.9)


def test_save_predictions_rejects_bad_shape(engine):
    exp = _fake_experiment(engine)
    with pytest.raises(ValueError):
        repo.save_predictions(engine, exp, "test", np.array([1, 2]), np.zeros((2, 5)))


def test_finish_experiment_rejects_unknown_field(engine):
    exp = _fake_experiment(engine)
    with pytest.raises(AttributeError):
        repo.finish_experiment(engine, exp, accuracy=1.0)
