"""Stage 3 tests: subject-grouped split, train-only standardization, data pipeline."""

import numpy as np
import pytest

from har.data.pipeline import prepare_data
from har.data.preprocess import ChannelStandardizer
from har.data.splits import subject_val_split


def test_val_split_holds_out_whole_subjects(synthetic_splits):
    subjects = synthetic_splits["train"].subjects
    tr, va, val_subj = subject_val_split(subjects, n_val_subjects=2, seed=0)
    assert len(val_subj) == 2
    assert not set(subjects[tr]) & set(subjects[va])  # no subject on both sides
    assert set(subjects[va]) == set(val_subj)
    assert len(tr) + len(va) == len(subjects)


def test_val_split_is_deterministic_per_seed(synthetic_splits):
    subjects = synthetic_splits["train"].subjects
    assert subject_val_split(subjects, 2, seed=7)[2] == subject_val_split(subjects, 2, seed=7)[2]


def test_val_split_rejects_bad_size(synthetic_splits):
    with pytest.raises(ValueError):
        subject_val_split(synthetic_splits["train"].subjects, n_val_subjects=0)


def test_standardizer_uses_train_statistics_only():
    rng = np.random.default_rng(0)
    train = rng.normal(5.0, 2.0, (50, 16, 3)).astype(np.float32)
    test = rng.normal(100.0, 1.0, (10, 16, 3)).astype(np.float32)
    sc = ChannelStandardizer().fit(train)
    z = sc.transform(train)
    np.testing.assert_allclose(z.mean(axis=(0, 1)), 0, atol=1e-4)
    np.testing.assert_allclose(z.std(axis=(0, 1)), 1, atol=1e-3)
    # test data must NOT be re-centred: it keeps its large offset relative to train
    assert sc.transform(test).mean() > 10


def test_standardizer_save_load_roundtrip(tmp_path):
    x = np.random.default_rng(1).normal(size=(8, 4, 9)).astype(np.float32)
    sc = ChannelStandardizer().fit(x)
    sc.save(tmp_path / "scaler.json")
    np.testing.assert_allclose(ChannelStandardizer.load(tmp_path / "scaler.json").transform(x),
                               sc.transform(x))


def test_standardizer_requires_fit():
    with pytest.raises(RuntimeError):
        ChannelStandardizer().transform(np.zeros((1, 2, 3), dtype=np.float32))


@pytest.mark.parametrize("rep,shape", [("features", (561,)), ("windows", (128, 9))])
def test_prepare_data_shapes_and_alignment(synthetic_splits, rep, shape):
    tr, te = synthetic_splits["train"], synthetic_splits["test"]
    d = prepare_data(rep, n_val_subjects=2, train=tr, test=te)
    assert d.X_train.shape[1:] == d.X_val.shape[1:] == d.X_test.shape[1:] == shape
    assert len(d.X_train) == len(d.y_train) == len(d.train_rows)
    np.testing.assert_array_equal(d.y_val, tr.labels[d.val_rows])
    np.testing.assert_array_equal(d.y_test, te.labels)
    assert not set(tr.subjects[d.train_rows]) & set(d.val_subjects)


def test_both_representations_use_identical_rows(synthetic_splits):
    tr, te = synthetic_splits["train"], synthetic_splits["test"]
    f = prepare_data("features", n_val_subjects=2, train=tr, test=te)
    w = prepare_data("windows", n_val_subjects=2, train=tr, test=te)
    np.testing.assert_array_equal(f.train_rows, w.train_rows)
    np.testing.assert_array_equal(f.val_rows, w.val_rows)
    assert f.val_subjects == w.val_subjects


def test_windows_standardizer_fitted_on_train_rows_only(synthetic_splits):
    tr, te = synthetic_splits["train"], synthetic_splits["test"]
    d = prepare_data("windows", n_val_subjects=2, train=tr, test=te)
    expected = tr.windows[d.train_rows].mean(axis=(0, 1))
    np.testing.assert_allclose(d.standardizer.mean_, expected, rtol=1e-5)


def test_unknown_representation_rejected(synthetic_splits):
    with pytest.raises(ValueError):
        prepare_data("images", train=synthetic_splits["train"], test=synthetic_splits["test"])
