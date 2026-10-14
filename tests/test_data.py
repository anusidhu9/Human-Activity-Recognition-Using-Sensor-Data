"""Stage 2 tests: loaders on a tiny synthetic copy of the dataset layout, plus
checks on the real dataset that are skipped when it has not been downloaded."""

import io
import zipfile

import numpy as np
import pytest

from har.config import INERTIAL_CHANNELS, NUM_FEATURES, PATHS, WINDOW_LENGTH
from har.data.download import _safe_extract, extract_dataset
from har.data.loaders import load_cached_split, load_feature_names, load_split

N = 5


def _write_fake_split(root, split, n=N, seed=0):
    rng = np.random.default_rng(seed)
    d = root / split
    (d / "Inertial Signals").mkdir(parents=True)
    np.savetxt(d / f"X_{split}.txt", rng.uniform(-1, 1, (n, NUM_FEATURES)))
    np.savetxt(d / f"y_{split}.txt", rng.integers(1, 7, n), fmt="%d")
    np.savetxt(d / f"subject_{split}.txt", np.full(n, 1 if split == "train" else 2), fmt="%d")
    for i, ch in enumerate(INERTIAL_CHANNELS):
        np.savetxt(d / "Inertial Signals" / f"{ch}_{split}.txt",
                   np.full((n, WINDOW_LENGTH), float(i)))


@pytest.fixture
def fake_dataset(tmp_path):
    root = tmp_path / "UCI HAR Dataset"
    _write_fake_split(root, "train")
    _write_fake_split(root, "test")
    (root / "features.txt").write_text("1 a\n2 b\n3 a\n")
    return root


def test_load_split_shapes_and_channel_order(fake_dataset):
    s = load_split("train", fake_dataset)
    assert s.features.shape == (N, NUM_FEATURES)
    assert s.windows.shape == (N, WINDOW_LENGTH, len(INERTIAL_CHANNELS))
    # channel i was filled with the value i, so the last axis must follow INERTIAL_CHANNELS
    np.testing.assert_array_equal(s.windows[0, 0], np.arange(len(INERTIAL_CHANNELS)))
    assert len(s) == N


def test_load_split_rejects_misaligned_files(fake_dataset):
    np.savetxt(fake_dataset / "train" / "y_train.txt", np.ones(N + 1), fmt="%d")
    with pytest.raises(ValueError):
        load_split("train", fake_dataset)


def test_unknown_split_rejected(fake_dataset):
    with pytest.raises(ValueError):
        load_split("validation", fake_dataset)


def test_duplicate_feature_names_are_suffixed(fake_dataset):
    assert load_feature_names(fake_dataset) == ["a", "b", "a__2"]


def test_cache_roundtrip_and_no_overwrite(fake_dataset, tmp_path):
    cache_dir = tmp_path / "processed"
    first = load_cached_split("train", fake_dataset, cache_dir)
    mtime = (cache_dir / "train.npz").stat().st_mtime_ns
    second = load_cached_split("train", fake_dataset, cache_dir)
    np.testing.assert_array_equal(first.windows, second.windows)
    assert (cache_dir / "train.npz").stat().st_mtime_ns == mtime


def test_safe_extract_blocks_path_traversal(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with zipfile.ZipFile(buf) as zf, pytest.raises(ValueError):
        _safe_extract(zf, tmp_path / "out")


def test_extract_skips_existing_dataset(tmp_path):
    (tmp_path / "UCI HAR Dataset").mkdir()
    marker = tmp_path / "UCI HAR Dataset" / "keep.txt"
    marker.write_text("original")
    extract_dataset(tmp_path / "missing.zip", tmp_path)  # must not even open the archive
    assert marker.read_text() == "original"


# ------------------------------------------------------------- real dataset checks
real = pytest.mark.skipif(not PATHS.dataset_dir.exists(), reason="UCI HAR dataset not downloaded")


@pytest.mark.dataset
@real
def test_real_dataset_official_split():
    train, test = load_cached_split("train"), load_cached_split("test")
    assert len(train) == 7352 and len(test) == 2947
    assert len(set(train.subjects)) == 21 and len(set(test.subjects)) == 9
    assert not set(train.subjects) & set(test.subjects), "subject leakage between train and test"
    assert set(train.labels) == set(range(1, 7))
